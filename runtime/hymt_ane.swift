// Hunyuan-dense ANE serve runtime.
//
// Heavy compute is ANE-only: 7 stateful transformer layer shards plus
// 2 RMSNorm+tied-LM-head shards. Host work: token-id embedding lookup,
// RoPE/mask bookkeeping, argmax sampling.
//
// Compile:
//   swiftc -O local-artifacts/hymt_ane.swift \
//     -framework CoreML -framework Foundation \
//     -o local-artifacts/hymt_ane_runtime
//
// Run:
//   ./hymt_ane_runtime --meta local-artifacts/hymt_ane/hymt_runtime_meta.json \
//     --prompt-ids 120000 --max-new 50
//
//   ./hymt_ane_runtime --meta <path> --serve
//   # stdin: {"prompt_ids":[120000,...],"max_new":50}
//   # stdout: {"ok":true,"generated_ids":[...],"timing":{...}}
//
//   ./hymt_ane_runtime --meta <path> --unix-socket /tmp/hymt.sock
//   # daemon mode: each connecting client gets independent KV states;
//   # weights loaded once, shared across all sessions.

import CoreML
import Darwin
import Foundation

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

struct LayerSpec: Decodable {
    let start: Int
    let end: Int
    let path: String
}

struct LMHeadShardSpec: Decodable {
    let shardIdx: Int
    let vocabStart: Int
    let vocabEnd: Int
    let mlmodelc: String

    enum CodingKeys: String, CodingKey {
        case shardIdx = "shard_idx"
        case vocabStart = "vocab_start"
        case vocabEnd = "vocab_end"
        case mlmodelc
    }
}

struct HymtRuntimeMeta: Decodable {
    let modelFamily: String
    let dModel: Int
    let nHeads: Int
    let nKvHeads: Int
    let dHead: Int
    let ropeDim: Int?
    let vocabSize: Int
    let nLayers: Int
    let maxSeqLen: Int
    let ropeFreqBase: Double
    let eosTokenId: Int
    let bosTokenId: Int
    let embedBin: String
    let layers: [LayerSpec]
    let lmHeadShards: [LMHeadShardSpec]
    let rangedimTMax: Int?

    enum CodingKeys: String, CodingKey {
        case modelFamily = "model_family"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case nKvHeads = "n_kv_heads"
        case dHead = "d_head"
        case ropeDim = "rope_dim"
        case vocabSize = "vocab_size"
        case nLayers = "n_layers"
        case maxSeqLen = "max_seq_len"
        case ropeFreqBase = "rope_freq_base"
        case eosTokenId = "eos_token_id"
        case bosTokenId = "bos_token_id"
        case embedBin = "embed_bin"
        case layers
        case lmHeadShards = "lm_head_shards"
        case rangedimTMax = "rangedim_t_max"
    }
}

// ---------------------------------------------------------------------------
// Serve protocol types
// ---------------------------------------------------------------------------

struct HymtServeRequest: Decodable {
    let promptIds: [Int]
    let maxNew: Int?
    let temperature: Float?
    let topK: Int?
    let topP: Float?
    let repPenalty: Float?
    let profile: Bool?
    let reset: Bool?     // nil/true = fresh start; false = continue from cached KV state

    enum CodingKeys: String, CodingKey {
        case promptIds = "prompt_ids"
        case maxNew = "max_new"
        case temperature
        case topK = "top_k"
        case topP = "top_p"
        case repPenalty = "rep_penalty"
        case profile
        case reset
    }
}

struct HymtServeTiming: Encodable {
    let prefillTokens: Int
    let prefillSeconds: Double
    let decodeTokens: Int
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let forwardCalls: Int
    let forwardSeconds: Double
    let forwardTokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case prefillTokens = "prefill_tokens"
        case prefillSeconds = "prefill_s"
        case decodeTokens = "decode_tokens"
        case decodeSeconds = "decode_s"
        case decodeTokensPerSecond = "decode_tok_s"
        case forwardCalls = "forward_calls"
        case forwardSeconds = "forward_s"
        case forwardTokensPerSecond = "forward_tok_s"
    }
}

struct HymtServeProfile: Encodable {
    let calls: Int
    let embedSeconds: Double
    let ropeMaskSeconds: Double
    let layersSeconds: Double
    let headCopySeconds: Double
    let headPredictReduceSeconds: Double

    enum CodingKeys: String, CodingKey {
        case calls
        case embedSeconds = "embed_s"
        case ropeMaskSeconds = "rope_mask_s"
        case layersSeconds = "layers_s"
        case headCopySeconds = "head_copy_s"
        case headPredictReduceSeconds = "head_predict_reduce_s"
    }
}

struct HymtServeResponse: Encodable {
    let ok: Bool
    let generatedIds: [Int]?
    let cacheSeqLen: Int?    // KV cache fill level after this request
    let timing: HymtServeTiming?
    let profile: HymtServeProfile?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case generatedIds = "generated_ids"
        case cacheSeqLen = "cache_seq_len"
        case timing
        case profile
        case error
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

final class FP16BinaryFile {
    let data: Data
    let count: Int

    init(path: String, expectedCount: Int) throws {
        data = try Data(contentsOf: URL(fileURLWithPath: path))
        count = data.count / MemoryLayout<Float16>.size
        precondition(count == expectedCount, "\(path): got \(count) fp16 values, expected \(expectedCount)")
    }

    func writeRow(_ index: Int, dim: Int, into ptr: UnsafeMutablePointer<Float16>) {
        precondition(index >= 0 && index * dim + dim <= count, "token id \(index) out of embedding bounds")
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
            memcpy(ptr, src + index * dim, dim * MemoryLayout<Float16>.size)
        }
    }

    func writeRow(_ index: Int, dim: Int, into ptr: UnsafeMutablePointer<Float16>,
                  channelStride: Int, tokenStride: Int, tokenSlot: Int) {
        precondition(index >= 0 && index * dim + dim <= count, "token id \(index) out of embedding bounds (strided)")
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float16.self) + index * dim
            for channel in 0..<dim {
                ptr[channel * channelStride + tokenSlot * tokenStride] = src[channel]
            }
        }
    }
}

func resolvePath(_ relative: String, relativeTo metaPath: String) -> String {
    if relative.hasPrefix("/") { return relative }
    let base = (metaPath as NSString).deletingLastPathComponent
    return (base as NSString).appendingPathComponent(relative)
}

func fillRoPE(cosPtr: UnsafeMutablePointer<Float16>,
              sinPtr: UnsafeMutablePointer<Float16>,
              pos: Int, dHalf: Int, base: Double) {
    for j in 0..<dHalf {
        let inv = 1.0 / pow(base, Double(j) / Double(dHalf))
        let angle = Double(pos) * inv
        cosPtr[j] = Float16(cos(angle))
        sinPtr[j] = Float16(sin(angle))
    }
}

func copyFlatFloat16(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float16>, count: Int) {
    let srcPtr = src.dataPointer.assumingMemoryBound(to: Float16.self)
    let shape = src.shape.map { Int(truncating: $0) }
    let strides = src.strides.map { Int(truncating: $0) }
    if shape == [1, count, 1, 1] {
        let stride1 = strides[1]
        for i in 0..<count { dst[i] = srcPtr[i * stride1] }
    } else {
        for i in 0..<count { dst[i] = srcPtr[i] }
    }
}

func findNGramProposal(history: [Int], minN: Int, maxN: Int) -> (token: Int, ngramSize: Int, matchIndex: Int)? {
    let tokenCount = history.count
    if minN <= 0 || maxN < minN || tokenCount <= minN { return nil }
    let largestNGram = min(maxN, tokenCount - 1)
    if largestNGram < minN { return nil }
    for ngramSize in stride(from: largestNGram, through: minN, by: -1) {
        let suffixStart = tokenCount - ngramSize
        if suffixStart <= 0 { continue }
        for matchStart in stride(from: suffixStart - 1, through: 0, by: -1) {
            var matched = true
            for offset in 0..<ngramSize {
                if history[matchStart + offset] != history[suffixStart + offset] {
                    matched = false; break
                }
            }
            if matched { return (history[matchStart + ngramSize], ngramSize, matchStart) }
        }
    }
    return nil
}

func copyHiddenTokens(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float16>,
                      d: Int, batchTokens: Int, dstChannelStride: Int, dstTokenStride: Int) {
    let srcPtr = src.dataPointer.assumingMemoryBound(to: Float16.self)
    let strides = src.strides.map { Int(truncating: $0) }
    let srcChannelStride = strides[1]
    let srcTokenStride   = strides[2]
    for channel in 0..<d {
        for tokenSlot in 0..<batchTokens {
            dst[channel * dstChannelStride + tokenSlot * dstTokenStride] =
                srcPtr[channel * srcChannelStride + tokenSlot * srcTokenStride]
        }
    }
}

func validateLayerCoverage(_ layers: [LayerSpec], nLayers: Int) {
    precondition(!layers.isEmpty, "manifest has no layer shards")
    var expectedStart = 0
    for spec in layers {
        precondition(spec.start == expectedStart,
            "layer shard gap/overlap at \(expectedStart); got [\(spec.start),\(spec.end))")
        precondition(spec.end > spec.start, "empty shard [\(spec.start),\(spec.end))")
        expectedStart = spec.end
    }
    precondition(expectedStart == nLayers,
        "layer shards cover 0..<\(expectedStart), expected 0..<\(nLayers)")
}

func printStderr(_ msg: String) {
    var s = FileHandle.standardError
    s.write((msg + "\n").data(using: .utf8)!)
}

func encodeJSON<T: Encodable>(_ value: T) throws -> String {
    let data = try JSONEncoder().encode(value)
    return String(data: data, encoding: .utf8)!
}

// Temperature + top-k + top-p nucleus sampling with optional repetition penalty.
// logits: full vocab, un-scaled. penalty > 1.0 discourages previously seen tokens.
func sampleLogits(_ logits: [Float], temperature: Float, topK: Int, topP: Float,
                  penalty: Float = 1.0, seen: [Int] = []) -> Int {
    let vocab = logits.count
    guard vocab > 0 else { return 0 }
    // Apply repetition penalty before temperature scaling
    var penalized = logits
    if penalty != 1.0 {
        for tok in seen {
            if tok < vocab {
                if penalized[tok] > 0 { penalized[tok] /= penalty }
                else { penalized[tok] *= penalty }
            }
        }
    }
    // Greedy when temp == 0
    if temperature <= 0 || topK == 1 {
        return penalized.indices.max(by: { penalized[$0] < penalized[$1] }) ?? 0
    }
    // Build (idx, scaled_logit) sorted descending
    let scale = 1.0 / temperature
    var pairs = [(Int, Float)]()
    pairs.reserveCapacity(vocab)
    for i in 0..<vocab { pairs.append((i, penalized[i] * scale)) }
    pairs.sort { $0.1 > $1.1 }
    // Top-k filter
    let k = topK > 0 ? min(topK, vocab) : vocab
    let topPairs = Array(pairs.prefix(k))
    // Softmax
    let maxVal = topPairs[0].1
    var exps = topPairs.map { expf($0.1 - maxVal) }
    let sumExp = exps.reduce(0.0, +)
    guard sumExp > 0 else { return topPairs[0].0 }
    for j in 0..<exps.count { exps[j] /= sumExp }
    // Top-p nucleus cutoff
    var cumProb: Float = 0
    var cutoff = exps.count
    for j in 0..<exps.count {
        cumProb += exps[j]
        if cumProb >= topP { cutoff = j + 1; break }
    }
    // Weighted sample
    let r = Float.random(in: 0..<1)
    var cdf: Float = 0
    for j in 0..<cutoff {
        cdf += exps[j]
        if r < cdf { return topPairs[j].0 }
    }
    return topPairs[min(cutoff - 1, topPairs.count - 1)].0
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

@available(macOS 15.0, *)
final class HymtRuntime {
    let meta: HymtRuntimeMeta
    let embed: FP16BinaryFile
    let layerModels: [MLModel]
    let headModels: [MLModel]
    let headSpecs: [LMHeadShardSpec]
    let headQueue: DispatchQueue

    // Dimension shortcuts
    let d: Int
    let maxSeqLen: Int
    let ropeHalf: Int

    // Persistent MLMultiArray buffers (allocated once, mutated per step)
    let xArr:         MLMultiArray  // (1, d, 1, 1) current hidden state
    let cosArr:       MLMultiArray  // (1, ropeHalf, 1, 1)
    let sinArr:       MLMultiArray  // (1, ropeHalf, 1, 1)
    let attnMaskArr:  MLMultiArray  // (1, 1, 1, maxSeqLen) fp16
    let kvWriteMaskArr: MLMultiArray// (1, 1, 1, maxSeqLen) fp16
    let headInputArr: MLMultiArray  // (1, d, 1, 1) — head model input

    // Typed raw pointers into the buffers (no bounds check in hot path)
    let xPtr:           UnsafeMutablePointer<Float16>
    let cosPtr:         UnsafeMutablePointer<Float16>
    let sinPtr:         UnsafeMutablePointer<Float16>
    let attnMaskPtr:    UnsafeMutablePointer<Float16>
    let kvWriteMaskPtr: UnsafeMutablePointer<Float16>
    let headInputPtr:   UnsafeMutablePointer<Float16>

    // Feature providers (populated from the buffers, reused per step)
    let layerProvider: MLDictionaryFeatureProvider
    let headProvider:  MLDictionaryFeatureProvider

    // Profile accumulators
    var tEmbed = 0.0
    var tRopeMask = 0.0
    var tLayers = 0.0
    var tHeadCopy = 0.0
    var tHeadPredictReduce = 0.0
    var profileCalls = 0

    var traceTokens = false

    // Verifier (RangeDim T=2..4) buffers — allocated once, reused per call
    let verifierBatchTokens: Int
    let headInputStride1: Int
    let verifierXArr:              MLMultiArray
    let verifierXPtr:              UnsafeMutablePointer<Float16>
    let verifierXStride1:          Int
    let verifierXStride2:          Int
    let verifierCosArr:            MLMultiArray
    let verifierCosPtr:            UnsafeMutablePointer<Float16>
    let verifierCosStride0:        Int
    let verifierSinArr:            MLMultiArray
    let verifierSinPtr:            UnsafeMutablePointer<Float16>
    let verifierSinStride0:        Int
    let verifierAttnMaskArr:       MLMultiArray
    let verifierAttnMaskPtr:       UnsafeMutablePointer<Float16>
    let verifierAttnMaskStride2:   Int
    let verifierAttnMaskStride3:   Int
    let verifierKVWriteMaskArr:    MLMultiArray
    let verifierKVWriteMaskPtr:    UnsafeMutablePointer<Float16>
    let verifierKVWriteMaskStride2: Int
    let verifierKVWriteMaskStride3: Int
    let verifierLayerProvider:     MLDictionaryFeatureProvider

    // Speculative/ngram config — set by main() after init
    var useSpeculative  = false
    var useNgramProbe   = false
    var useNgramForce   = false
    var ngramMin        = 2
    var ngramMax        = 8
    // Per-request counters
    var specCalls = 0; var specDrafted = 0; var specAccepted = 0; var specFallbacks = 0
    var ngramTargets = 0; var ngramProposals = 0; var ngramAccepted_stat = 0
    var ngramProposalBySize = [Int: Int](); var ngramAcceptedBySize = [Int: Int]()

    init(meta: HymtRuntimeMeta, embed: FP16BinaryFile,
         layerModels: [MLModel], headModels: [MLModel], headSpecs: [LMHeadShardSpec]) throws {
        self.meta = meta
        self.embed = embed
        self.layerModels = layerModels
        self.headModels = headModels
        self.headSpecs = headSpecs
        self.headQueue = DispatchQueue(label: "hymt.head", attributes: .concurrent)

        d = meta.dModel
        maxSeqLen = meta.maxSeqLen
        ropeHalf = (meta.ropeDim ?? meta.dHead) / 2

        // Allocate buffers
        xArr         = try MLMultiArray(shape: [1, d, 1, 1] as [NSNumber], dataType: .float16)
        cosArr       = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber], dataType: .float16)
        sinArr       = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber], dataType: .float16)
        attnMaskArr  = try MLMultiArray(shape: [1, 1, 1, maxSeqLen] as [NSNumber], dataType: .float16)
        kvWriteMaskArr = try MLMultiArray(shape: [1, 1, maxSeqLen, 1] as [NSNumber], dataType: .float16)
        headInputArr = try MLMultiArray(shape: [1, d, 1, 1] as [NSNumber], dataType: .float16)

        // Raw typed pointers
        xPtr           = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
        cosPtr         = cosArr.dataPointer.assumingMemoryBound(to: Float16.self)
        sinPtr         = sinArr.dataPointer.assumingMemoryBound(to: Float16.self)
        attnMaskPtr    = attnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        kvWriteMaskPtr = kvWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        headInputPtr   = headInputArr.dataPointer.assumingMemoryBound(to: Float16.self)

        // Fill attn mask with -inf (all past positions masked by default)
        let negInf = Float16(-65504)  // fp16 max negative
        for i in 0..<maxSeqLen {
            attnMaskPtr[i]    = negInf
            kvWriteMaskPtr[i] = Float16(0.0)
        }

        // Build providers
        layerProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x":            MLFeatureValue(multiArray: xArr),
            "rope_cos":     MLFeatureValue(multiArray: cosArr),
            "rope_sin":     MLFeatureValue(multiArray: sinArr),
            "attn_mask":    MLFeatureValue(multiArray: attnMaskArr),
            "kv_write_mask":MLFeatureValue(multiArray: kvWriteMaskArr),
        ])
        headProvider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden": MLFeatureValue(multiArray: headInputArr),
        ])

        // ── Verifier (RangeDim T=2..4) buffers ─────────────────────────────
        headInputStride1   = Int(truncating: headInputArr.strides[1])
        verifierBatchTokens = meta.rangedimTMax ?? 4
        let vbt = verifierBatchTokens
        verifierXArr = try MLMultiArray(
            shape: [1, NSNumber(value: d), NSNumber(value: vbt), 1], dataType: .float16)
        verifierXPtr    = verifierXArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierXStride1 = Int(truncating: verifierXArr.strides[1])
        verifierXStride2 = Int(truncating: verifierXArr.strides[2])

        verifierCosArr = try MLMultiArray(
            shape: [NSNumber(value: vbt), NSNumber(value: ropeHalf)], dataType: .float16)
        verifierCosPtr    = verifierCosArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierCosStride0 = Int(truncating: verifierCosArr.strides[0])

        verifierSinArr = try MLMultiArray(
            shape: [NSNumber(value: vbt), NSNumber(value: ropeHalf)], dataType: .float16)
        verifierSinPtr    = verifierSinArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierSinStride0 = Int(truncating: verifierSinArr.strides[0])

        verifierAttnMaskArr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: vbt), NSNumber(value: maxSeqLen)], dataType: .float16)
        verifierAttnMaskPtr = verifierAttnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierAttnMaskStride2 = Int(truncating: verifierAttnMaskArr.strides[2])
        verifierAttnMaskStride3 = Int(truncating: verifierAttnMaskArr.strides[3])

        verifierKVWriteMaskArr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: maxSeqLen), NSNumber(value: vbt)], dataType: .float16)
        verifierKVWriteMaskPtr = verifierKVWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierKVWriteMaskStride2 = Int(truncating: verifierKVWriteMaskArr.strides[2])
        verifierKVWriteMaskStride3 = Int(truncating: verifierKVWriteMaskArr.strides[3])

        verifierLayerProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x":             MLFeatureValue(multiArray: verifierXArr),
            "rope_cos":      MLFeatureValue(multiArray: verifierCosArr),
            "rope_sin":      MLFeatureValue(multiArray: verifierSinArr),
            "attn_mask":     MLFeatureValue(multiArray: verifierAttnMaskArr),
            "kv_write_mask": MLFeatureValue(multiArray: verifierKVWriteMaskArr),
        ])
    }

    func makeStates() throws -> [MLState] {
        return try layerModels.map { try $0.makeState() }
    }

    func resetMasks() {
        let negInf = Float16(-65504)
        for i in 0..<maxSeqLen {
            attnMaskPtr[i]    = negInf
            kvWriteMaskPtr[i] = Float16(0.0)
        }
    }

    func resetVerifierInputs() {
        let negInf = Float16(-65504)
        for channel in 0..<d {
            for slot in 0..<verifierBatchTokens {
                verifierXPtr[channel * verifierXStride1 + slot * verifierXStride2] = 0
            }
        }
        for slot in 0..<verifierBatchTokens {
            for j in 0..<maxSeqLen {
                verifierAttnMaskPtr[slot * verifierAttnMaskStride2 + j * verifierAttnMaskStride3] = negInf
                verifierKVWriteMaskPtr[j * verifierKVWriteMaskStride2 + slot * verifierKVWriteMaskStride3] = 0
            }
        }
    }

    func fillVerifierRoPE(row: Int, pos: Int) {
        for j in 0..<ropeHalf {
            let inv = 1.0 / pow(meta.ropeFreqBase, Double(j) / Double(ropeHalf))
            let angle = Double(pos) * inv
            verifierCosPtr[row * verifierCosStride0 + j] = Float16(cos(angle))
            verifierSinPtr[row * verifierSinStride0 + j] = Float16(sin(angle))
        }
    }

    // Run the T=1 LM head for each of `count` token slots stored in verifierXArr.
    // Returns the argmax token for each slot (greedy decode only for HyMT).
    func predictSlotsWithT1Head(count: Int) throws -> [Int] {
        var results = [Int](repeating: -1, count: count)
        for slot in 0..<count {
            // Copy slot's hidden state into the T=1 head input buffer
            for channel in 0..<d {
                headInputPtr[channel * headInputStride1] =
                    verifierXPtr[channel * verifierXStride1 + slot * verifierXStride2]
            }
            let group = DispatchGroup()
            let lock  = NSLock()
            var shardBestLogit = [Float](repeating: -Float.infinity, count: headModels.count)
            var shardBestToken = [Int](repeating: -1, count: headModels.count)
            var shardError: Error? = nil
            for s in 0..<headModels.count {
                group.enter()
                headQueue.async {
                    do {
                        let result = try self.headModels[s].prediction(from: self.headProvider)
                        let arr    = result.featureValue(for: "logits")!.multiArrayValue!
                        let ptr    = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                        let stride1 = Int(truncating: arr.strides[1])
                        let spec   = self.headSpecs[s]
                        var localBestLogit = -Float.infinity
                        var localBestToken = -1
                        for local in 0..<(spec.vocabEnd - spec.vocabStart) {
                            let value = Float(ptr[local * stride1])
                            if value > localBestLogit { localBestLogit = value; localBestToken = spec.vocabStart + local }
                        }
                        lock.lock()
                        shardBestLogit[s] = localBestLogit
                        shardBestToken[s] = localBestToken
                        lock.unlock()
                    } catch {
                        lock.lock(); if shardError == nil { shardError = error }; lock.unlock()
                    }
                    group.leave()
                }
            }
            group.wait()
            if let error = shardError { throw error }
            var bestLogit = -Float.infinity
            var bestToken = -1
            for s in 0..<headModels.count {
                if shardBestToken[s] >= 0 && shardBestLogit[s] > bestLogit {
                    bestLogit = shardBestLogit[s]; bestToken = shardBestToken[s]
                }
            }
            if bestToken < 0 {
                throw NSError(domain: "HymtANE", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "T=1 head: no token for slot \(slot)"])
            }
            results[slot] = bestToken
        }
        return results
    }

    // Forward `tokens.count` tokens (T=2..4) through verifier (RangeDim) shards.
    // When advanceOnly=true, skip LM head — only KV cache is written.
    func forwardVerifier(tokens: [Int], posStart: Int, cacheSeqLen: Int,
                         states: [MLState], collectProfile: Bool = false,
                         advanceOnly: Bool = false) throws -> [Int] {
        precondition(tokens.count >= 1 && tokens.count <= verifierBatchTokens)
        let t0 = CFAbsoluteTimeGetCurrent()
        resetVerifierInputs()
        for slot in 0..<verifierBatchTokens {
            fillVerifierRoPE(row: slot, pos: posStart + slot)
        }
        for slot in 0..<tokens.count {
            embed.writeRow(tokens[slot], dim: d, into: verifierXPtr,
                           channelStride: verifierXStride1, tokenStride: verifierXStride2,
                           tokenSlot: slot)
            let visibleEnd = min(maxSeqLen - 1, cacheSeqLen + slot)
            if visibleEnd >= 0 {
                for j in 0...visibleEnd {
                    verifierAttnMaskPtr[slot * verifierAttnMaskStride2 + j * verifierAttnMaskStride3] = 0
                }
            }
            verifierKVWriteMaskPtr[(cacheSeqLen + slot) * verifierKVWriteMaskStride2
                                   + slot * verifierKVWriteMaskStride3] = Float16(1.0)
        }
        let t1 = CFAbsoluteTimeGetCurrent()
        for idx in 0..<layerModels.count {
            try autoreleasepool {
                let result = try layerModels[idx].prediction(from: verifierLayerProvider, using: states[idx])
                let output = result.featureValue(for: "hidden")!.multiArrayValue!
                copyHiddenTokens(output, into: verifierXPtr, d: d,
                                 batchTokens: verifierBatchTokens,
                                 dstChannelStride: verifierXStride1,
                                 dstTokenStride: verifierXStride2)
            }
        }
        let t2 = CFAbsoluteTimeGetCurrent()
        if advanceOnly {
            if collectProfile { tEmbed += t1 - t0; tLayers += t2 - t1; profileCalls += 1 }
            return []
        }
        let predictions = try predictSlotsWithT1Head(count: tokens.count)
        let t3 = CFAbsoluteTimeGetCurrent()
        if collectProfile { tEmbed += t1 - t0; tLayers += t2 - t1; tHeadPredictReduce += t3 - t2; profileCalls += 1 }
        return predictions
    }

    func speculativeDraft(history: [Int], firstToken: Int) -> [Int] {
        var draft = [firstToken]
        var scratch = history
        while draft.count < verifierBatchTokens {
            guard let proposal = findNGramProposal(history: scratch, minN: ngramMin, maxN: ngramMax) else { break }
            draft.append(proposal.token)
            scratch.append(proposal.token)
        }
        return draft
    }

    func runGenerationSpeculative(promptIds: [Int], maxNew: Int, states: [MLState],
                                  requestProfile: Bool) throws
        -> (generated: [Int], cacheSeqLen: Int, timing: HymtServeTiming, profile: HymtServeProfile?) {
        precondition(!promptIds.isEmpty)
        precondition(promptIds.count + maxNew <= meta.maxSeqLen)
        specCalls = 0; specDrafted = 0; specAccepted = 0; specFallbacks = 0
        var generated = [Int]()
        var cacheSeqLen = 0
        var forwardCalls = 0
        var next: Int = -1

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var i = 0
        while i < promptIds.count {
            let chunkEnd = min(i + verifierBatchTokens, promptIds.count)
            let chunk = Array(promptIds[i..<chunkEnd])
            let isLast = (chunkEnd == promptIds.count)
            if isLast {
                let preds = try forwardVerifier(tokens: chunk, posStart: cacheSeqLen,
                                                cacheSeqLen: cacheSeqLen, states: states)
                next = preds[chunk.count - 1]
            } else {
                _ = try forwardVerifier(tokens: chunk, posStart: cacheSeqLen,
                                        cacheSeqLen: cacheSeqLen, states: states, advanceOnly: true)
            }
            cacheSeqLen += chunk.count
            forwardCalls += 1
            i += chunk.count
        }
        let prefillElapsed = CFAbsoluteTimeGetCurrent() - prefillStart
        generated.append(next)
        if requestProfile { tEmbed = 0; tLayers = 0; tHeadPredictReduce = 0; profileCalls = 0 }

        // Prime T=1 attn mask for any forwardOne fallback calls
        for j in 0..<cacheSeqLen { attnMaskPtr[j] = 0 }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        while generated.count < maxNew {
            let history = promptIds + generated
            let draft = speculativeDraft(history: history, firstToken: generated.last!)
            let predictions = try forwardVerifier(tokens: draft,
                posStart: promptIds.count + generated.count - 1,
                cacheSeqLen: cacheSeqLen, states: states, collectProfile: requestProfile)
            forwardCalls += 1; specCalls += 1
            specDrafted += max(0, draft.count - 1)

            var acceptedContinuation = 0
            var emittedFallback = false
            if draft.count > 1 {
                for idx in 1..<draft.count {
                    if predictions[idx - 1] == draft[idx] {
                        generated.append(draft[idx])
                        acceptedContinuation += 1; specAccepted += 1
                        if generated.count >= maxNew { break }
                    } else {
                        generated.append(predictions[idx - 1])
                        emittedFallback = true; specFallbacks += 1
                        break
                    }
                }
            }
            if !emittedFallback && generated.count < maxNew {
                generated.append(predictions[draft.count - 1])
            }
            cacheSeqLen += 1 + acceptedContinuation
            if isEOS(generated.last!) { break }
        }
        let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
        let decodeTokens = max(0, generated.count - 1)
        let timing = HymtServeTiming(
            prefillTokens: promptIds.count,
            prefillSeconds: prefillElapsed,
            decodeTokens: decodeTokens,
            decodeSeconds: decodeElapsed,
            decodeTokensPerSecond: decodeElapsed > 0 && decodeTokens > 0 ? Double(decodeTokens) / decodeElapsed : 0,
            forwardCalls: forwardCalls,
            forwardSeconds: prefillElapsed + decodeElapsed,
            forwardTokensPerSecond: (prefillElapsed + decodeElapsed) > 0 ? Double(forwardCalls) / (prefillElapsed + decodeElapsed) : 0
        )
        let profile = requestProfile && profileCalls > 0 ? HymtServeProfile(
            calls: profileCalls, embedSeconds: tEmbed, ropeMaskSeconds: tRopeMask,
            layersSeconds: tLayers, headCopySeconds: tHeadCopy,
            headPredictReduceSeconds: tHeadPredictReduce) : nil
        if requestProfile { tEmbed = 0; tRopeMask = 0; tLayers = 0; tHeadCopy = 0; tHeadPredictReduce = 0; profileCalls = 0 }
        return (generated, cacheSeqLen, timing, profile)
    }

    func isEOS(_ token: Int) -> Bool {
        token == meta.eosTokenId || token == 120001 || token == 120008
    }

    func forwardOne(tokenId: Int, pos: Int, cacheSeqLen: inout Int,
                    states: [MLState], collectProfile: Bool = false,
                    skipHead: Bool = false,
                    temperature: Float = 1.0, topK: Int = 0, topP: Float = 1.0,
                    repPenalty: Float = 1.0, seen: [Int] = []) throws -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        embed.writeRow(tokenId, dim: d, into: xPtr)
        let t1 = CFAbsoluteTimeGetCurrent()

        fillRoPE(cosPtr: cosPtr, sinPtr: sinPtr, pos: pos, dHalf: ropeHalf, base: meta.ropeFreqBase)
        attnMaskPtr[cacheSeqLen] = 0
        if cacheSeqLen > 0 { kvWriteMaskPtr[cacheSeqLen - 1] = 0 }
        kvWriteMaskPtr[cacheSeqLen] = Float16(1.0)
        let t2 = CFAbsoluteTimeGetCurrent()

        for idx in 0..<layerModels.count {
            let result = try layerModels[idx].prediction(from: layerProvider, using: states[idx])
            let output = result.featureValue(for: "hidden")!.multiArrayValue!
            copyFlatFloat16(output, into: xPtr, count: d)
        }
        let t3 = CFAbsoluteTimeGetCurrent()
        cacheSeqLen += 1

        // Skip head during prefill — logits are discarded anyway
        if skipHead {
            if collectProfile {
                tEmbed    += t1 - t0
                tRopeMask += t2 - t1
                tLayers   += t3 - t2
                profileCalls += 1
            }
            return -1
        }

        // Copy hidden → head input
        memcpy(headInputPtr, xPtr, d * MemoryLayout<Float16>.size)
        let t4 = CFAbsoluteTimeGetCurrent()

        // Concurrent head dispatch — collect full logits from each shard
        let group = DispatchGroup()
        let lock  = NSLock()
        var shardLogits = [[Float]](repeating: [], count: headModels.count)
        var shardError: Error? = nil

        for s in 0..<headModels.count {
            group.enter()
            headQueue.async {
                do {
                    let result = try self.headModels[s].prediction(from: self.headProvider)
                    let arr    = result.featureValue(for: "logits")!.multiArrayValue!
                    let ptr    = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                    let stride1 = Int(truncating: arr.strides[1])
                    let spec   = self.headSpecs[s]
                    let count  = spec.vocabEnd - spec.vocabStart
                    var floats = [Float](repeating: 0, count: count)
                    for local in 0..<count { floats[local] = Float(ptr[local * stride1]) }
                    lock.lock()
                    shardLogits[s] = floats
                    lock.unlock()
                } catch {
                    lock.lock()
                    if shardError == nil { shardError = error }
                    lock.unlock()
                }
                group.leave()
            }
        }
        group.wait()
        if let error = shardError { throw error }
        let t5 = CFAbsoluteTimeGetCurrent()

        // Merge shard logits and sample
        var allLogits = [Float]()
        allLogits.reserveCapacity(meta.vocabSize)
        for s in 0..<headModels.count { allLogits.append(contentsOf: shardLogits[s]) }
        let bestToken = sampleLogits(allLogits, temperature: temperature, topK: topK, topP: topP,
                                     penalty: repPenalty, seen: seen)

        if collectProfile {
            tEmbed             += t1 - t0
            tRopeMask          += t2 - t1
            tLayers            += t3 - t2
            tHeadCopy          += t4 - t3
            tHeadPredictReduce += t5 - t4
            profileCalls       += 1
        }
        if traceTokens {
            print("pos=\(pos) in=\(tokenId) next=\(bestToken)")
        }
        return bestToken
    }

    func runGeneration(promptIds: [Int], maxNew: Int, states: [MLState],
                       startCacheSeqLen: Int = 0,
                       requestProfile: Bool, temperature: Float = 1.0,
                       topK: Int = 0, topP: Float = 1.0, repPenalty: Float = 1.0) throws -> (generated: [Int], cacheSeqLen: Int, timing: HymtServeTiming, profile: HymtServeProfile?) {
        precondition(!promptIds.isEmpty)
        precondition(startCacheSeqLen + promptIds.count + maxNew <= meta.maxSeqLen,
            "cache (\(startCacheSeqLen)) + prompt (\(promptIds.count)) + max_new (\(maxNew)) exceeds max_seq_len (\(meta.maxSeqLen))")

        if startCacheSeqLen == 0 { resetMasks() }

        // Route to speculative generation when flag is set
        if useSpeculative {
            let (gen, endCache, timing, profile) = try runGenerationSpeculative(
                promptIds: promptIds, maxNew: maxNew, states: states, requestProfile: requestProfile)
            return (gen, startCacheSeqLen + endCache, timing, profile)
        }

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var cacheSeqLen = startCacheSeqLen
        var forwardCalls = 0
        var nextToken: Int = -1
        var usedBatchPrefill = false

        // Chunked prefill: process up to verifierBatchTokens per ANE call
        var prefillIdx = 0
        while prefillIdx < promptIds.count {
            let chunkEnd = min(prefillIdx + verifierBatchTokens, promptIds.count)
            let chunk = Array(promptIds[prefillIdx..<chunkEnd])
            let isLast = (chunkEnd == promptIds.count)
            if chunk.count == 1 {
                let token = chunk[0]
                if isLast {
                    nextToken = try forwardOne(tokenId: token, pos: cacheSeqLen,
                                              cacheSeqLen: &cacheSeqLen, states: states,
                                              collectProfile: requestProfile)
                } else {
                    _ = try forwardOne(tokenId: token, pos: cacheSeqLen,
                                       cacheSeqLen: &cacheSeqLen, states: states,
                                       skipHead: true)
                }
            } else {
                usedBatchPrefill = true
                if isLast {
                    let preds = try forwardVerifier(tokens: chunk, posStart: cacheSeqLen,
                                                   cacheSeqLen: cacheSeqLen, states: states,
                                                   collectProfile: requestProfile)
                    nextToken = preds[chunk.count - 1]
                } else {
                    _ = try forwardVerifier(tokens: chunk, posStart: cacheSeqLen,
                                            cacheSeqLen: cacheSeqLen, states: states,
                                            advanceOnly: true)
                }
                cacheSeqLen += chunk.count
            }
            forwardCalls += 1
            prefillIdx += chunk.count
        }
        // Prime T=1 attn mask after any batch prefill
        if usedBatchPrefill {
            for j in 0..<cacheSeqLen { attnMaskPtr[j] = 0 }
        }
        let prefillEnd = CFAbsoluteTimeGetCurrent()

        // Ngram probe: measure how often n-gram proposal matches actual next token
        if useNgramProbe {
            ngramTargets += 1
            if let proposal = findNGramProposal(history: promptIds, minN: ngramMin, maxN: ngramMax) {
                ngramProposals += 1
                ngramProposalBySize[proposal.ngramSize, default: 0] += 1
                if proposal.token == nextToken {
                    ngramAccepted_stat += 1
                    ngramAcceptedBySize[proposal.ngramSize, default: 0] += 1
                }
            }
        }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generated = [Int]()
        var seenTokens = promptIds
        generated.append(nextToken)
        seenTokens.append(nextToken)

        while generated.count < maxNew && !isEOS(nextToken) {
            let history = seenTokens
            // Ngram force: override sampling with n-gram proposal when available
            if useNgramForce,
               let proposal = findNGramProposal(history: history, minN: ngramMin, maxN: ngramMax) {
                nextToken = proposal.token
                cacheSeqLen += 1  // Skip forward pass — directly use forced token
                generated.append(nextToken); seenTokens.append(nextToken)
                continue
            }
            nextToken = try forwardOne(tokenId: generated.last!, pos: cacheSeqLen,
                                       cacheSeqLen: &cacheSeqLen, states: states,
                                       collectProfile: requestProfile,
                                       temperature: temperature, topK: topK, topP: topP,
                                       repPenalty: repPenalty, seen: seenTokens)
            generated.append(nextToken)
            seenTokens.append(nextToken)
            forwardCalls += 1
            if useNgramProbe {
                ngramTargets += 1
                if let proposal = findNGramProposal(history: Array(seenTokens.dropLast()),
                                                   minN: ngramMin, maxN: ngramMax) {
                    ngramProposals += 1
                    ngramProposalBySize[proposal.ngramSize, default: 0] += 1
                    if proposal.token == nextToken {
                        ngramAccepted_stat += 1
                        ngramAcceptedBySize[proposal.ngramSize, default: 0] += 1
                    }
                }
            }
        }
        let decodeEnd = CFAbsoluteTimeGetCurrent()

        let prefillSec = prefillEnd - prefillStart
        let decodeSec  = decodeEnd - decodeStart
        let forwardSec = prefillSec + decodeSec
        let decodeTokens = max(0, generated.count - 1)
        let timing = HymtServeTiming(
            prefillTokens: promptIds.count,
            prefillSeconds: prefillSec,
            decodeTokens: decodeTokens,
            decodeSeconds: decodeSec,
            decodeTokensPerSecond: decodeTokens > 0 && decodeSec > 0 ? Double(decodeTokens) / decodeSec : 0,
            forwardCalls: forwardCalls,
            forwardSeconds: forwardSec,
            forwardTokensPerSecond: forwardSec > 0 ? Double(forwardCalls) / forwardSec : 0
        )
        var profile: HymtServeProfile? = nil
        if requestProfile && profileCalls > 0 {
            profile = HymtServeProfile(
                calls: profileCalls,
                embedSeconds: tEmbed,
                ropeMaskSeconds: tRopeMask,
                layersSeconds: tLayers,
                headCopySeconds: tHeadCopy,
                headPredictReduceSeconds: tHeadPredictReduce
            )
            tEmbed = 0; tRopeMask = 0; tLayers = 0; tHeadCopy = 0; tHeadPredictReduce = 0
            profileCalls = 0
        }
        return (generated, cacheSeqLen, timing, profile)
    }
}

// ---------------------------------------------------------------------------
// Unix socket daemon helpers
// ---------------------------------------------------------------------------

/// Line-buffered reader over a raw file descriptor.
struct FDLineReader {
    let fd: Int32
    var buffer: [UInt8] = []

    mutating func readLine() -> String? {
        while true {
            if let nlIdx = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let lineBytes = Array(buffer[..<nlIdx])
                buffer = Array(buffer[(nlIdx + 1)...])
                return String(bytes: lineBytes, encoding: .utf8)
            }
            var chunk = [UInt8](repeating: 0, count: 4096)
            let n = Darwin.read(fd, &chunk, 4096)
            if n <= 0 {
                if !buffer.isEmpty {
                    let line = String(bytes: buffer, encoding: .utf8)
                    buffer = []
                    return line
                }
                return nil
            }
            buffer.append(contentsOf: chunk[..<n])
        }
    }
}

func writeLineFD(_ fd: Int32, _ s: String) {
    var bytes = Array((s + "\n").utf8)
    _ = Darwin.write(fd, &bytes, bytes.count)
}

@available(macOS 15.0, *)
func runSocketServer(socketPath: String, runtime: HymtRuntime,
                     warmupCalls: Int, warmupIds: [Int]) throws {
    let serverFd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard serverFd >= 0 else {
        throw NSError(domain: "HymtSocket", code: Int(errno),
                      userInfo: [NSLocalizedDescriptionKey: "socket() failed: \(String(cString: strerror(errno)))"])
    }
    var reuse: Int32 = 1
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
    unlink(socketPath)

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    guard socketPath.utf8.count < MemoryLayout.size(ofValue: addr.sun_path) else {
        throw NSError(domain: "HymtSocket", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Socket path too long"])
    }
    withUnsafeMutableBytes(of: &addr.sun_path) { buf in
        socketPath.withCString { src in
            _ = strcpy(buf.baseAddress!.assumingMemoryBound(to: CChar.self), src)
        }
    }
    let bindResult = withUnsafePointer(to: addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.bind(serverFd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard bindResult == 0 else {
        Darwin.close(serverFd)
        throw NSError(domain: "HymtSocket", code: Int(errno),
                      userInfo: [NSLocalizedDescriptionKey: "bind() failed: \(String(cString: strerror(errno)))"])
    }
    Darwin.listen(serverFd, 8)

    print("READY {\"protocol\":\"hymt-jsonl-v1\",\"transport\":\"unix-socket\",\"path\":\"\(socketPath)\"}")
    fflush(stdout)
    printStderr("Listening on \(socketPath)")

    // Accept loop — serial (one client at a time; ANE is a single resource anyway)
    while true {
        let clientFd = Darwin.accept(serverFd, nil, nil)
        guard clientFd >= 0 else { continue }
        printStderr("Client connected")

        // Each connection gets its own fresh KV state
        guard let sessionStates = try? runtime.makeStates() else {
            Darwin.close(clientFd); continue
        }
        // Per-startup warmup already done; reset cache for this session
        var sessionCacheSeqLen = 0

        var reader = FDLineReader(fd: clientFd)
        while let line = reader.readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            do {
                let req = try JSONDecoder().decode(HymtServeRequest.self, from: Data(trimmed.utf8))
                if req.reset ?? true { sessionCacheSeqLen = 0 }
                let (generated, endCacheSeqLen, timing, prof) = try runtime.runGeneration(
                    promptIds: req.promptIds,
                    maxNew: req.maxNew ?? 50,
                    states: sessionStates,
                    startCacheSeqLen: sessionCacheSeqLen,
                    requestProfile: req.profile ?? false,
                    temperature: req.temperature ?? 1.0,
                    topK: req.topK ?? 0,
                    topP: req.topP ?? 1.0,
                    repPenalty: req.repPenalty ?? 1.0)
                sessionCacheSeqLen = endCacheSeqLen
                let resp = HymtServeResponse(ok: true, generatedIds: generated,
                                             cacheSeqLen: endCacheSeqLen,
                                             timing: timing, profile: prof, error: nil)
                writeLineFD(clientFd, try encodeJSON(resp))
            } catch {
                let resp = HymtServeResponse(ok: false, generatedIds: nil,
                                             cacheSeqLen: nil, timing: nil, profile: nil,
                                             error: error.localizedDescription)
                if let s = try? encodeJSON(resp) { writeLineFD(clientFd, s) }
            }
        }
        Darwin.close(clientFd)
        printStderr("Client disconnected")
    }
}

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------

func main() throws {
    var metaPath   = "local-artifacts/hymt_ane/hymt_runtime_meta.json"
    var promptIds  = [120000]
    var maxNew     = 1
    var warmupCalls = 0
    var traceTokens = false
    var profile    = false
    var serve      = false
    var speculative = false
    var ngramProbe  = false
    var ngramForce  = false
    var ngramMin    = 2
    var ngramMax    = 8
    var unixSocketPath: String? = nil
    var temperature: Float = 1.0
    var topK: Int = 0
    var topP: Float = 1.0
    var repPenalty: Float = 1.0
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--meta":        metaPath = args[i + 1]; i += 2
        case "--prompt-ids":  promptIds = args[i + 1].split(separator: ",")
                              .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }; i += 2
        case "--max-new":     maxNew = Int(args[i + 1])!; i += 2
        case "--warmup":      warmupCalls = Int(args[i + 1])!; i += 2
        case "--temperature": temperature = Float(args[i + 1])!; i += 2
        case "--top-k":       topK = Int(args[i + 1])!; i += 2
        case "--top-p":       topP = Float(args[i + 1])!; i += 2
        case "--rep-pen":     repPenalty = Float(args[i + 1])!; i += 2
        case "--serve":        serve = true; i += 1
        case "--unix-socket":  unixSocketPath = args[i + 1]; i += 2
        case "--trace":        traceTokens = true; i += 1
        case "--profile":      profile = true; i += 1
        case "--speculative":  speculative = true; i += 1
        case "--ngram-probe":  ngramProbe = true; i += 1
        case "--ngram-force":  ngramForce = true; i += 1
        case "--ngram-min":    ngramMin = Int(args[i + 1])!; i += 2
        case "--ngram-max":    ngramMax = Int(args[i + 1])!; i += 2
        default:               i += 1
        }
    }

    let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    let meta     = try JSONDecoder().decode(HymtRuntimeMeta.self, from: metaData)
    let sortedLayers = meta.layers.sorted { $0.start < $1.start }
    validateLayerCoverage(sortedLayers, nLayers: meta.nLayers)
    precondition(!meta.lmHeadShards.isEmpty, "manifest has no lm_head_shards")

    func status(_ msg: String) {
        if serve { printStderr(msg) } else { print(msg) }
    }

    status("Loading \(meta.modelFamily): \(meta.nLayers)L d=\(meta.dModel) vocab=\(meta.vocabSize) layer_shards=\(sortedLayers.count) head_shards=\(meta.lmHeadShards.count)")

    let embedPath = resolvePath(meta.embedBin, relativeTo: metaPath)
    let embed = try FP16BinaryFile(path: embedPath, expectedCount: meta.vocabSize * meta.dModel)
    status("Embed: \(meta.vocabSize * meta.dModel * 2 / 1024 / 1024) MB loaded from \(embedPath)")

    guard #available(macOS 15.0, *) else {
        fputs("ERROR: stateful CoreML models require macOS 15+\n", stderr); exit(1)
    }

    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine

    status("Loading layer shards...")
    let layerModels: [MLModel] = try sortedLayers.map { spec in
        let path = resolvePath(spec.path, relativeTo: metaPath)
        let t0 = CFAbsoluteTimeGetCurrent()
        let m = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        status("  [\(spec.start),\(spec.end)) \(String(format: "%.2f", dt))s")
        return m
    }

    let sortedHeadSpecs = meta.lmHeadShards.sorted { $0.vocabStart < $1.vocabStart }
    status("Loading LM-head shards...")
    let headModels: [MLModel] = try sortedHeadSpecs.map { spec in
        let path = resolvePath(spec.mlmodelc, relativeTo: metaPath)
        let t0 = CFAbsoluteTimeGetCurrent()
        let m = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        status("  vocab[\(spec.vocabStart),\(spec.vocabEnd)) \(String(format: "%.2f", dt))s")
        return m
    }

    precondition(!(ngramProbe && ngramForce), "--ngram-probe and --ngram-force are mutually exclusive")
    precondition(!(speculative && ngramForce), "--speculative and --ngram-force are mutually exclusive")

    let runtime = try HymtRuntime(meta: meta, embed: embed,
                                   layerModels: layerModels, headModels: headModels,
                                   headSpecs: sortedHeadSpecs)
    runtime.traceTokens    = traceTokens
    runtime.useSpeculative = speculative
    runtime.useNgramProbe  = ngramProbe
    runtime.useNgramForce  = ngramForce
    runtime.ngramMin       = ngramMin
    runtime.ngramMax       = ngramMax

    status("Allocating states...")
    let states = try runtime.makeStates()
    status("Ready.")

    // Warmup
    if warmupCalls > 0 {
        status("Warming up (\(warmupCalls) call(s))...")
        for _ in 0..<warmupCalls {
            _ = try runtime.runGeneration(promptIds: [meta.bosTokenId], maxNew: 1,
                                           states: states, requestProfile: false)
        }
        status("Warmup done.")
    }

    if let sockPath = unixSocketPath {
        try runSocketServer(socketPath: sockPath, runtime: runtime,
                            warmupCalls: warmupCalls, warmupIds: [meta.bosTokenId])
    } else if serve {
        // Serve mode: read JSON lines from stdin, write JSON lines to stdout
        // Signal readiness on stdout for the Python client to detect.
        print("READY {\"protocol\":\"hymt-jsonl-v1\"}")
        fflush(stdout)
        // Persistent state: reused across requests; reset only when req.reset != false
        var serveCacheSeqLen = 0
        while let line = readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            do {
                let req = try JSONDecoder().decode(
                    HymtServeRequest.self,
                    from: Data(trimmed.utf8))
                // reset: nil or true → start fresh; false → continue from cached KV
                if req.reset ?? true { serveCacheSeqLen = 0 }
                let (generated, endCacheSeqLen, timing, prof) = try runtime.runGeneration(
                    promptIds: req.promptIds,
                    maxNew: req.maxNew ?? 50,
                    states: states,
                    startCacheSeqLen: serveCacheSeqLen,
                    requestProfile: req.profile ?? false,
                    temperature: req.temperature ?? 1.0,
                    topK: req.topK ?? 0,
                    topP: req.topP ?? 1.0,
                    repPenalty: req.repPenalty ?? 1.0)
                serveCacheSeqLen = endCacheSeqLen
                let resp = HymtServeResponse(ok: true, generatedIds: generated,
                                             cacheSeqLen: endCacheSeqLen,
                                             timing: timing, profile: prof, error: nil)
                print(try encodeJSON(resp))
            } catch {
                let resp = HymtServeResponse(ok: false, generatedIds: nil,
                                             cacheSeqLen: nil,
                                             timing: nil, profile: nil,
                                             error: error.localizedDescription)
                print(try encodeJSON(resp))
            }
            fflush(stdout)
        }
    } else {
        // Single-shot mode
        precondition(!promptIds.isEmpty, "--prompt-ids must not be empty")
        let (generated, _, timing, prof) = try runtime.runGeneration(
            promptIds: promptIds, maxNew: maxNew, states: states, requestProfile: profile,
            temperature: temperature, topK: topK, topP: topP, repPenalty: repPenalty)
        print("Generated IDs: \(generated)")
        print(String(format: "Prefill: %d tok in %.3fs", timing.prefillTokens, timing.prefillSeconds))
        print(String(format: "Decode:  %d tok in %.3fs → %.1f tok/s",
                     timing.decodeTokens, timing.decodeSeconds, timing.decodeTokensPerSecond))
        if let p = prof {
            print(String(format: "Profile: calls=%d embed=%.3fs rope=%.3fs layers=%.3fs head=%.3fs",
                         p.calls, p.embedSeconds, p.ropeMaskSeconds, p.layersSeconds,
                         p.headCopySeconds + p.headPredictReduceSeconds))
        }
    }
}

try main()
