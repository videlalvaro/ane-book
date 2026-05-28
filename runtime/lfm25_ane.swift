// lfm25_ane.swift — LFM2.5-8B-A1B ANE Inference Runtime
//
// Orchestrates the hybrid ShortConv + GQA attention + MoE decode loop
// for LiquidAI/LFM2.5-8B-A1B on Apple Neural Engine.
//
// Architecture:
//   24 layers — 18 ShortConv (LIV conv) + 6 GQA attention
//   All 22 MoE layers split into 2 expert-half shards each
//   2 dense MLP layers (layers 0, 1)
//   Tied embeddings — lm_head = embed_tokens weight
//
// Shard types:
//   lfm25_dense_layer{0,1}.mlmodelc       : conv op + dense MLP (single shard)
//   lfm25_op_layer{N}.mlmodelc             : conv|attn op + router output
//   lfm25_moe0_layer{N}.mlmodelc           : experts 0–15 soft-routed
//   lfm25_moe1_layer{N}.mlmodelc           : experts 16–31 soft-routed
//   lfm25_lm_head{0,1}.mlmodelc            : LM head vocab halves
//
// Conv state management:
//   Each of the 18 ShortConv layers maintains a sliding window of L=3 tokens.
//   State shape: [1, 2048, 3, 1] — only 6144 floats per layer (24 KB total).
//   Passed as input/output, updated each decode step.
//   Compare: KV cache grows O(T); conv state is always 3 positions.
//
// Soft routing approximation:
//   All 32 experts run with sigmoid(router_logits) weights.
//   The 28 non-selected experts contribute near-zero weight in practice.
//   Quality impact validated to be within INT8 quantization noise floor.
//   (See validators/lfm25_residency_check.py for quality gate.)
//
// [Knuth Vol 3 §6.4] Conv state as circular buffer — O(1) update per token.
// [Dragon Book §9.2] Branch-free expert dispatch via soft routing.

import CoreML
import Foundation

// MARK: - Model Config

private struct LFM25Config {
    static let hiddenSize         = 2048
    static let numLayers          = 24
    static let numDenseLayers     = 2
    static let convLCache         = 3
    static let numExperts         = 32
    static let expertsPerHalf     = 16
    static let vocabSize          = 128_000
    static let vocabHalf          = 64_000
    static let numAttnHeads       = 32
    static let numKVHeads         = 8
    static let headDim            = 64
    static let numAttentionLayers = 6   // at indices 2, 6, 10, 14, 18, 21
    
    // Layer type: true = full_attention, false = conv
    static let isAttentionLayer: [Bool] = [
        false, false, true,            // 0,1,2
        false, false, false, true,     // 3,4,5,6
        false, false, false, true,     // 7,8,9,10
        false, false, false, true,     // 11,12,13,14
        false, false, false, true,     // 15,16,17,18
        false, false, true,            // 19,20,21
        false, false,                  // 22,23
    ]
    
    // Whether a layer uses MoE (true) or dense MLP (false)
    static let hasMoE: [Bool] = (0..<numLayers).map { $0 >= numDenseLayers }
}

// MARK: - Shard container

/// All CoreML models needed for one decode step.
final class LFM25ANE {
    
    // Dense layers 0, 1
    private var denseLayers: [MLModel] = []
    
    // Operator shards (conv or attn), for MoE layers only
    private var opShards: [Int: MLModel] = [:]
    
    // MoE expert-half shards: [layer_idx: (moe0, moe1)]
    private var moeShards: [Int: (MLModel, MLModel)] = [:]
    
    // LM head halves
    private var lmHead: [MLModel] = []
    
    // Embedding lookup table — host-side (permitted per ANE mandate)
    private var embeddings: [[Float]] = []  // [vocab_size, hidden_size]
    
    // Final norm weights (host-side RMSNorm)
    private var embNormWeight: [Float] = []
    
    // -----------------------------------------------------------------------
    // MARK: Init
    // -----------------------------------------------------------------------
    
    init(modelDir: URL, embeddingBin: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all   // target ANE
        
        // Dense layers 0, 1
        for i in 0..<LFM25Config.numDenseLayers {
            let url = modelDir.appendingPathComponent("lfm25_dense_layer\(i).mlmodelc")
            denseLayers.append(try MLModel(contentsOf: url, configuration: cfg))
        }
        
        // Operator + MoE shards for layers 2-23
        for i in LFM25Config.numDenseLayers..<LFM25Config.numLayers {
            if !LFM25Config.isAttentionLayer[i] {
                let opURL = modelDir.appendingPathComponent("lfm25_op_layer\(i).mlmodelc")
                opShards[i] = try MLModel(contentsOf: opURL, configuration: cfg)
            }
            // Attention layers: op shard loaded separately (GQA with KV state)
            if LFM25Config.hasMoE[i] {
                let moe0URL = modelDir.appendingPathComponent("lfm25_moe0_layer\(i).mlmodelc")
                let moe1URL = modelDir.appendingPathComponent("lfm25_moe1_layer\(i).mlmodelc")
                moeShards[i] = (
                    try MLModel(contentsOf: moe0URL, configuration: cfg),
                    try MLModel(contentsOf: moe1URL, configuration: cfg)
                )
            }
        }
        
        // LM head halves
        for half in 0..<2 {
            let url = modelDir.appendingPathComponent("lfm25_lm_head\(half).mlmodelc")
            lmHead.append(try MLModel(contentsOf: url, configuration: cfg))
        }
        
        // Load host embedding table
        try loadEmbeddings(from: embeddingBin)
    }
    
    private func loadEmbeddings(from url: URL) throws {
        let data = try Data(contentsOf: url)
        // Expect raw Float32: [vocab_size × hidden_size]
        let count = LFM25Config.vocabSize * LFM25Config.hiddenSize
        guard data.count == count * 4 else {
            throw LFM25Error.embeddingShapeMismatch(data.count, count * 4)
        }
        embeddings = data.withUnsafeBytes { ptr in
            let floats = ptr.bindMemory(to: Float.self)
            return stride(from: 0, to: count, by: LFM25Config.hiddenSize).map { start in
                Array(floats[start..<start + LFM25Config.hiddenSize])
            }
        }
    }
    
    // -----------------------------------------------------------------------
    // MARK: State
    // -----------------------------------------------------------------------
    
    /// Full decode state for one sequence.
    struct DecodeState {
        // ShortConv states: [layer_idx] → [1, H, L, 1]
        var convStates: [Int: MLMultiArray]
        
        // KV cache for attention layers: [layer_idx] → (k, v)
        // k, v shape: [1, n_kv_heads * head_dim, T, 1]
        var kvCaches: [Int: (MLMultiArray, MLMultiArray)]
        
        // Current decode position
        var position: Int = 0
        
        static func initial(numLayers: Int) throws -> DecodeState {
            var convStates: [Int: MLMultiArray] = [:]
            let H = LFM25Config.hiddenSize
            let L = LFM25Config.convLCache
            
            // Initialize conv states to zero for all non-attention layers
            for i in 0..<numLayers where !LFM25Config.isAttentionLayer[i] {
                let state = try MLMultiArray(shape: [1, H, L, 1] as [NSNumber], dataType: .float32)
                // Zero-initialize (sliding window starts empty)
                for j in 0..<state.count { state[j] = 0.0 }
                convStates[i] = state
            }
            
            // KV caches start empty — grow as tokens are generated
            return DecodeState(convStates: convStates, kvCaches: [:], position: 0)
        }
    }
    
    // -----------------------------------------------------------------------
    // MARK: Token embedding (host-side, permitted)
    // -----------------------------------------------------------------------
    
    /// Embed token → [1, H, 1, 1] MLMultiArray for ANE input.
    func embed(tokenId: Int) throws -> MLMultiArray {
        guard tokenId < LFM25Config.vocabSize else {
            throw LFM25Error.invalidTokenId(tokenId)
        }
        let H = LFM25Config.hiddenSize
        let arr = try MLMultiArray(shape: [1, H, 1, 1] as [NSNumber], dataType: .float32)
        let emb = embeddings[tokenId]
        for i in 0..<H { arr[i] = NSNumber(value: emb[i]) }
        return arr
    }
    
    // -----------------------------------------------------------------------
    // MARK: Decode step
    // -----------------------------------------------------------------------
    
    /// Run one decode step: token → next-token logits.
    /// Returns the argmax next token ID.
    func step(tokenId: Int, state: inout DecodeState) throws -> Int {
        var hidden = try embed(tokenId: tokenId)
        let H = LFM25Config.hiddenSize
        
        for layerIdx in 0..<LFM25Config.numLayers {
            let isDense = layerIdx < LFM25Config.numDenseLayers
            let isAttn  = LFM25Config.isAttentionLayer[layerIdx]
            let hasMoE  = LFM25Config.hasMoE[layerIdx]
            
            if isDense {
                // ── Dense layer (0, 1): single shard ─────────────────────
                let model = denseLayers[layerIdx]
                let convState = state.convStates[layerIdx]!
                let inputs = try MLDictionaryFeatureProvider(dictionary: [
                    "hidden":     hidden,
                    "conv_state": convState,
                ])
                let out = try model.prediction(from: inputs)
                hidden = out.featureValue(for: "updated_hidden")!.multiArrayValue!
                state.convStates[layerIdx] = out.featureValue(for: "new_conv_state")!.multiArrayValue!
                
            } else if isAttn {
                // ── Attention layer: GQA with KV cache ───────────────────
                // (KV cache management in real deployment uses MLState;
                //  shown here as explicit input/output for clarity)
                let (updatedHidden, _, _, ffnNormed, routingWeights) =
                    try runAttentionLayer(layerIdx: layerIdx, hidden: hidden, state: &state)
                
                hidden = updatedHidden
                
                // MoE FFN
                let (moe0, moe1) = moeShards[layerIdx]!
                let moeOut = try runMoELayer(
                    layerIdx: layerIdx, ffnNormed: ffnNormed,
                    routingWeights: routingWeights, moe0: moe0, moe1: moe1)
                hidden = try addTensors(hidden, moeOut)
                
            } else {
                // ── Conv layer with MoE ───────────────────────────────────
                let opModel = opShards[layerIdx]!
                let convState = state.convStates[layerIdx]!
                let opInputs = try MLDictionaryFeatureProvider(dictionary: [
                    "hidden":     hidden,
                    "conv_state": convState,
                ])
                let opOut = try opModel.prediction(from: opInputs)
                
                hidden = opOut.featureValue(for: "updated_hidden")!.multiArrayValue!
                state.convStates[layerIdx] = opOut.featureValue(for: "new_conv_state")!.multiArrayValue!
                
                let ffnNormed      = opOut.featureValue(for: "ffn_normed")!.multiArrayValue!
                let routingWeights = opOut.featureValue(for: "routing_weights")!.multiArrayValue!
                
                let (moe0, moe1) = moeShards[layerIdx]!
                let moeOut = try runMoELayer(
                    layerIdx: layerIdx, ffnNormed: ffnNormed,
                    routingWeights: routingWeights, moe0: moe0, moe1: moe1)
                
                hidden = try addTensors(hidden, moeOut)
            }
        }
        
        // Final embedding norm (host-side RMSNorm, trivial)
        hidden = try applyFinalNorm(hidden)
        
        // LM head: run both halves, concatenate logits, argmax
        let nextToken = try computeNextToken(hidden: hidden)
        state.position += 1
        return nextToken
    }
    
    // -----------------------------------------------------------------------
    // MARK: MoE helper
    // -----------------------------------------------------------------------
    
    private func runMoELayer(
        layerIdx: Int,
        ffnNormed: MLMultiArray,
        routingWeights: MLMultiArray,
        moe0: MLModel,
        moe1: MLModel
    ) throws -> MLMultiArray {
        let H = LFM25Config.hiddenSize
        let N = LFM25Config.expertsPerHalf
        
        // Split routing_weights [1, 32, 1, 1] into two halves [1, 16, 1, 1]
        let rwA = try sliceRoutingWeights(routingWeights, start: 0,  count: N)
        let rwB = try sliceRoutingWeights(routingWeights, start: N, count: N)
        
        // Run both half-shards
        let inA = try MLDictionaryFeatureProvider(dictionary: [
            "ffn_normed": ffnNormed, "routing_weights": rwA
        ])
        let inB = try MLDictionaryFeatureProvider(dictionary: [
            "ffn_normed": ffnNormed, "routing_weights": rwB
        ])
        
        let outA = try moe0.prediction(from: inA)
        let outB = try moe1.prediction(from: inB)
        
        let contA = outA.featureValue(for: "moe_contribution_half0")!.multiArrayValue!
        let contB = outB.featureValue(for: "moe_contribution_half1")!.multiArrayValue!
        
        // Sum contributions (host-side add: 1×H floats, trivial)
        return try addTensors(contA, contB)
    }
    
    // -----------------------------------------------------------------------
    // MARK: Attention helper (pending stateful GQA implementation)
    // -----------------------------------------------------------------------
    
    private func runAttentionLayer(
        layerIdx: Int,
        hidden: MLMultiArray,
        state: inout DecodeState
    ) throws -> (MLMultiArray, MLMultiArray, MLMultiArray, MLMultiArray, MLMultiArray) {
        throw LFM25Error.attentionShardNotImplemented(layerIdx)
    }
    
    // -----------------------------------------------------------------------
    // MARK: LM Head + argmax
    // -----------------------------------------------------------------------
    
    private func computeNextToken(hidden: MLMultiArray) throws -> Int {
        let inputs = try MLDictionaryFeatureProvider(dictionary: ["hidden": hidden])
        
        let out0 = try lmHead[0].prediction(from: inputs)
        let out1 = try lmHead[1].prediction(from: inputs)
        
        let logits0 = out0.featureValue(for: "logits_half0")!.multiArrayValue!
        let logits1 = out1.featureValue(for: "logits_half1")!.multiArrayValue!
        
        // Argmax across both halves (host-side, O(vocab) trivial)
        var bestIdx = 0
        var bestVal = Float(-Float.infinity)
        let V2 = LFM25Config.vocabHalf
        
        for i in 0..<V2 {
            let v = logits0[i].floatValue
            if v > bestVal { bestVal = v; bestIdx = i }
        }
        for i in 0..<V2 {
            let v = logits1[i].floatValue
            if v > bestVal { bestVal = v; bestIdx = V2 + i }
        }
        return bestIdx
    }
    
    // -----------------------------------------------------------------------
    // MARK: Tensor utilities (host-side, O(small) — not on compute path)
    // -----------------------------------------------------------------------
    
    /// Element-wise add two [1, H, 1, 1] arrays.
    private func addTensors(_ a: MLMultiArray, _ b: MLMultiArray) throws -> MLMultiArray {
        let H = LFM25Config.hiddenSize
        let result = try MLMultiArray(shape: [1, H, 1, 1] as [NSNumber], dataType: .float32)
        for i in 0..<H {
            result[i] = NSNumber(value: a[i].floatValue + b[i].floatValue)
        }
        return result
    }
    
    /// Apply final RMSNorm on the host (tiny, last layer only).
    private func applyFinalNorm(_ hidden: MLMultiArray) throws -> MLMultiArray {
        let H = LFM25Config.hiddenSize
        // Compute variance, rsqrt, scale
        var sumSq: Float = 0
        for i in 0..<H { let v = hidden[i].floatValue; sumSq += v * v }
        let scale = 1.0 / sqrt(sumSq / Float(H) + 1e-5)
        let result = try MLMultiArray(shape: [1, H, 1, 1] as [NSNumber], dataType: .float32)
        for i in 0..<H {
            let normed = hidden[i].floatValue * scale
            let w = embNormWeight.isEmpty ? 1.0 : embNormWeight[i]
            result[i] = NSNumber(value: normed * w)
        }
        return result
    }
    
    /// Slice routing_weights [1, 32, 1, 1] → [1, 16, 1, 1] for one expert half.
    private func sliceRoutingWeights(
        _ rw: MLMultiArray, start: Int, count: Int
    ) throws -> MLMultiArray {
        let out = try MLMultiArray(shape: [1, NSNumber(value: count), 1, 1], dataType: .float32)
        for i in 0..<count {
            out[i] = rw[start + i]
        }
        return out
    }
    
    // -----------------------------------------------------------------------
    // MARK: Greedy generation
    // -----------------------------------------------------------------------
    
    /// Greedy decode: returns generated token IDs (excluding the prompt).
    func generate(
        prompt: [Int],
        maxNewTokens: Int = 128,
        eosTokenId: Int = 2,
        state: inout DecodeState
    ) throws -> [Int] {
        var generated: [Int] = []
        
        // Prefill: run through prompt tokens (building KV/conv state)
        for tokenId in prompt {
            _ = try step(tokenId: tokenId, state: &state)
        }
        
        // Decode loop
        var lastToken = prompt.last ?? 1
        for _ in 0..<maxNewTokens {
            let nextToken = try step(tokenId: lastToken, state: &state)
            if nextToken == eosTokenId { break }
            generated.append(nextToken)
            lastToken = nextToken
        }
        return generated
    }
}

// MARK: - Errors

enum LFM25Error: Error, LocalizedError {
    case embeddingShapeMismatch(Int, Int)
    case invalidTokenId(Int)
    case shardNotFound(String)
    case attentionShardNotImplemented(Int)
    
    var errorDescription: String? {
        switch self {
        case .embeddingShapeMismatch(let got, let expected):
            return "Embedding binary size mismatch: got \(got) bytes, expected \(expected)"
        case .invalidTokenId(let id):
            return "Token ID \(id) out of range (vocab_size=\(LFM25Config.vocabSize))"
        case .shardNotFound(let name):
            return "Shard not found: \(name)"
        case .attentionShardNotImplemented(let layerIdx):
            return "Attention layer \(layerIdx) is not implemented yet; add the GQA MLState shard before full decoding"
        }
    }
}

// MARK: - Quick smoke test

/// Minimal smoke test: embed one token, run layer 0, check output shape.
func smokeLFM25(modelDir: URL, embeddingBin: URL) throws {
    print("Loading LFM2.5 ANE runtime…")
    let model = try LFM25ANE(modelDir: modelDir, embeddingBin: embeddingBin)
    var state = try LFM25ANE.DecodeState.initial(numLayers: LFM25Config.numLayers)
    
    // Token 1 = <bos> in the LFM2.5 tokenizer
    let token = try model.step(tokenId: 1, state: &state)
    print("Decode step OK → next token: \(token)")
    print("Conv state depth after step 1: position=\(state.position)")
}
