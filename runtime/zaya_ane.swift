// ZAYA1-8B ANE serve runtime.
//
// All 80 transformer layers (40 attn + 40 MoE) run entirely on ANE.
// Host work: embedding lookup, sampling, JSON I/O.
//
// Book references applied throughout:
//   [EoP §2]  Stepanov & McJones, Elements of Programming §2 — allocate all
//             buffers once; zero-alloc hot path; typed raw pointer access.
//   [APL]     Iverson, A Programming Language — the 3-shard LM head is a
//             parallel outer-product reduction: each shard independently
//             computes vocab[start..end] × hidden, then results are concatenated.
//   [DB §8]   Dragon Book §8.7 (peephole) — prefill optimisation: skip the
//             LM head during all but the final prompt token; those logits
//             are discarded anyway, saving ~3 concurrent model invocations
//             per prefill token.
//   [HyMT]    Lessons from the HyMT runtime: MLDictionaryFeatureProvider
//             created once holding mutable MLMultiArray references; CoreML
//             reads from the provider's array each call, so mutating xPtr
//             in-place between layer calls requires no new allocation.
//
// Architecture (non-stateful probe shards):
//   Even layers 0,2,...,78 → attn shard:  hidden → attn_out
//   Odd  layers 1,3,...,79 → MoE  shard:  hidden → moe_out
//   LM head 3 shards:                     hidden → logits (vocab partition)
//
// NOTE: These shards implement a SIMPLIFIED attention (Q→O projection,
// no KV cache). They are probe shards that validate full ANE execution and
// measure throughput. A future rebuild with stateful shards will add KV cache.
//
// Compile:
//   swiftc -O runtime/zaya_ane.swift \
//     -framework CoreML -framework Foundation \
//     -o runtime/zaya_ane_runtime
//
// Single-shot:
//   ./zaya_ane_runtime --meta models/zaya/zaya_runtime_meta.json \
//     --prompt-ids 2,42 --max-new 20 --profile
//
// Serve (stdin JSON lines → stdout JSON lines):
//   ./zaya_ane_runtime --meta models/zaya/zaya_runtime_meta.json --serve
//   # stdin:  {"prompt_ids":[2,42],"max_new":50}
//   # stdout: {"ok":true,"generated_ids":[...],"timing":{...}}
//
// Unix socket daemon (weights loaded once, per-connection independent state):
//   ./zaya_ane_runtime --meta ... --unix-socket /tmp/zaya.sock

import CoreML
import Darwin
import Foundation

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

struct ZayaLayerSpec: Decodable {
    let layer: Int
    let kind: String          // "attn" | "moe"
    let mlmodelc: String
}

struct ZayaLMHeadShardSpec: Decodable {
    let shardIdx: Int
    let vocabStart: Int
    let vocabEnd: Int
    let mlmodelc: String

    enum CodingKeys: String, CodingKey {
        case shardIdx    = "shard_idx"
        case vocabStart  = "vocab_start"
        case vocabEnd    = "vocab_end"
        case mlmodelc
    }
}

struct ZayaRuntimeMeta: Decodable {
    let modelFamily: String
    let modelId: String
    let dModel: Int
    let vocabSize: Int
    let nLayers: Int
    let nAttnHeads: Int
    let nKvHeads: Int
    let dHead: Int
    let nExperts: Int
    let embedBin: String
    let bosTokenId: Int
    let eosTokenId: Int
    let maxSeqLen: Int
    let layers: [ZayaLayerSpec]
    let lmHeadShards: [ZayaLMHeadShardSpec]
    // Stateful manifest fields (optional — absent in probe manifests)
    let attnImplementation: String?
    let rangedimTMax: Int?
    let moeRangedim: Bool?

    enum CodingKeys: String, CodingKey {
        case modelFamily = "model_family"
        case modelId     = "model_id"
        case dModel      = "d_model"
        case vocabSize   = "vocab_size"
        case nLayers     = "n_layers"
        case nAttnHeads  = "n_attn_heads"
        case nKvHeads    = "n_kv_heads"
        case dHead       = "d_head"
        case nExperts    = "n_experts"
        case embedBin    = "embed_bin"
        case bosTokenId  = "bos_token_id"
        case eosTokenId  = "eos_token_id"
        case maxSeqLen   = "max_seq_len"
        case layers
        case lmHeadShards    = "lm_head_shards"
        case attnImplementation = "attn_implementation"
        case rangedimTMax    = "rangedim_t_max"
        case moeRangedim     = "moe_rangedim"
    }
}

// ---------------------------------------------------------------------------
// Serve protocol types
// ---------------------------------------------------------------------------

struct ZayaServeRequest: Decodable {
    let promptIds: [Int]
    let maxNew: Int?
    let temperature: Float?
    let topK: Int?
    let topP: Float?
    let repPenalty: Float?
    let profile: Bool?

    enum CodingKeys: String, CodingKey {
        case promptIds   = "prompt_ids"
        case maxNew      = "max_new"
        case temperature
        case topK        = "top_k"
        case topP        = "top_p"
        case repPenalty  = "rep_penalty"
        case profile
    }
}

struct ZayaServeTiming: Encodable {
    let prefillTokens: Int
    let prefillSeconds: Double
    let decodeTokens: Int
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let forwardCalls: Int
    let forwardSeconds: Double
    let forwardTokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case prefillTokens          = "prefill_tokens"
        case prefillSeconds         = "prefill_s"
        case decodeTokens           = "decode_tokens"
        case decodeSeconds          = "decode_s"
        case decodeTokensPerSecond  = "decode_tok_s"
        case forwardCalls           = "forward_calls"
        case forwardSeconds         = "forward_s"
        case forwardTokensPerSecond = "forward_tok_s"
    }
}

struct ZayaServeProfile: Encodable {
    let calls: Int
    let embedSeconds: Double
    let layersSeconds: Double
    let headSeconds: Double

    enum CodingKeys: String, CodingKey {
        case calls
        case embedSeconds  = "embed_s"
        case layersSeconds = "layers_s"
        case headSeconds   = "head_s"
    }
}

struct ZayaServeResponse: Encodable {
    let ok: Bool
    let generatedIds: [Int]?
    let timing: ZayaServeTiming?
    let profile: ZayaServeProfile?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case generatedIds = "generated_ids"
        case timing
        case profile
        case error
    }
}

// ---------------------------------------------------------------------------
// FP16BinaryFile — zero-copy row lookup [EoP §2]
// ---------------------------------------------------------------------------

/// Memory-maps the embedding binary.  writeRow() copies one row into a caller-
/// provided buffer; no allocation on the hot path.
final class FP16BinaryFile {
    let data: Data
    let count: Int

    init(path: String, expectedCount: Int) throws {
        // Use NSData mapped (read-only) to avoid loading 1 GB into RAM eagerly.
        let url = URL(fileURLWithPath: path)
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        count = data.count / MemoryLayout<Float16>.size
        precondition(count == expectedCount,
            "\(path): expected \(expectedCount) fp16 values, got \(count)")
    }

    /// Copy embedding row `index` (d = `dim` values) into caller-owned buffer.
    @inline(__always)
    func writeRow(_ index: Int, dim: Int, into dst: UnsafeMutablePointer<Float16>) {
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
            memcpy(dst, src + index * dim, dim * MemoryLayout<Float16>.size)
        }
    }

    /// Write embedding row `index` into column `tokenSlot` of a strided buffer.
    /// Destination has shape [1, dim, batchT, 1] → channelStride = batchT, tokenStride = 1.
    /// Used by the T=vbt verifier to write each draft token into verifierXArr. [EoP §2]
    @inline(__always)
    func writeRow(_ index: Int, dim: Int, into dst: UnsafeMutablePointer<Float16>,
                  channelStride: Int, tokenSlot: Int) {
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float16.self) + index * dim
            for ch in 0..<dim {
                dst[ch * channelStride + tokenSlot] = src[ch]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

func resolvePath(_ relative: String, relativeTo metaPath: String) -> String {
    if relative.hasPrefix("/") { return relative }
    let base = (metaPath as NSString).deletingLastPathComponent
    return (base as NSString).appendingPathComponent(relative)
}

/// Copy a (1,d,1,1) or (1,d) MLMultiArray into a flat Float16 buffer.
/// Handles non-unit strides — no extra allocation. [EoP §2]
@inline(__always)
func copyFlatFloat16(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float16>, count: Int) {
    let srcPtr  = src.dataPointer.assumingMemoryBound(to: Float16.self)
    let strides = src.strides.map { Int(truncating: $0) }
    // Fast path: contiguous layout (stride-1 in the channel dim)
    if strides.last == 1 && strides.count >= 2 && strides[1] == 1 {
        memcpy(dst, srcPtr, count * MemoryLayout<Float16>.size)
    } else {
        // (1, count, 1, 1) with stride[1] stride
        let s = strides.count > 1 ? strides[1] : 1
        for i in 0..<count { dst[i] = srcPtr[i * s] }
    }
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
// [Concrete Math Ch.9 — complexity of this is O(V) with single-pass max then
// O(k log k) sort; cheap vs the ANE forward pass dominates runtime.]
func sampleLogits(_ logits: [Float], temperature: Float, topK: Int, topP: Float,
                  penalty: Float = 1.0, seen: [Int] = []) -> Int {
    let vocab = logits.count
    guard vocab > 0 else { return 0 }
    var penalized = logits
    if penalty != 1.0 {
        for tok in seen {
            if tok < vocab {
                if penalized[tok] > 0 { penalized[tok] /= penalty }
                else                  { penalized[tok] *= penalty }
            }
        }
    }
    if temperature <= 0 || topK == 1 {
        return penalized.indices.max(by: { penalized[$0] < penalized[$1] }) ?? 0
    }
    let scale = 1.0 / temperature
    var pairs = [(Int, Float)]()
    pairs.reserveCapacity(min(topK > 0 ? topK : vocab, vocab))
    for i in 0..<vocab { pairs.append((i, penalized[i] * scale)) }
    pairs.sort { $0.1 > $1.1 }
    let k = topK > 0 ? min(topK, vocab) : vocab
    let topPairs = Array(pairs.prefix(k))
    let maxVal   = topPairs[0].1
    var exps     = topPairs.map { expf($0.1 - maxVal) }
    let sumExp   = exps.reduce(0.0, +)
    guard sumExp > 0 else { return topPairs[0].0 }
    for j in 0..<exps.count { exps[j] /= sumExp }
    var cumProb: Float = 0
    var cutoff = exps.count
    for j in 0..<exps.count {
        cumProb += exps[j]
        if cumProb >= topP { cutoff = j + 1; break }
    }
    let r = Float.random(in: 0..<1)
    var cdf: Float = 0
    for j in 0..<cutoff {
        cdf += exps[j]
        if r < cdf { return topPairs[j].0 }
    }
    return topPairs[min(cutoff - 1, topPairs.count - 1)].0
}

// ---------------------------------------------------------------------------
// Speculative-decode helpers (architecture-agnostic; ported from HyMT Exp28)
// ---------------------------------------------------------------------------

/// Longest-suffix n-gram lookup.  Returns the token that historically followed
/// the current suffix of `history`, along with the matching n-gram size.
/// [Concrete Math Ch.9] — linear scan; cheap vs ANE forward pass.
func findNGramProposal(history: [Int], minN: Int, maxN: Int)
    -> (token: Int, ngramSize: Int, matchIndex: Int)? {
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

/// Copy a T-batch hidden state [1,d,T,1] into a flat strided destination.
/// Called after each T=vbt attn layer to refresh verifierXArr in-place.
@inline(__always)
func copyHiddenTokens(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float16>,
                      d: Int, batchTokens: Int,
                      dstChannelStride: Int, dstTokenStride: Int) {
    let srcPtr = src.dataPointer.assumingMemoryBound(to: Float16.self)
    let strides = src.strides.map { Int(truncating: $0) }
    let srcChStride  = strides.count > 1 ? strides[1] : 1
    let srcTokStride = strides.count > 2 ? strides[2] : 1
    for ch in 0..<d {
        for tok in 0..<batchTokens {
            dst[ch * dstChannelStride + tok * dstTokenStride] =
                srcPtr[ch * srcChStride + tok * srcTokStride]
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE table precomputation (for stateful attn shards)
// ZAYA: rope_theta=5000000, partial_rotary_factor=0.5 → rope_half=32
// [BOOK_ANALYSIS Exp30] — precompute once at startup, index by position.
// ---------------------------------------------------------------------------

/// Build cos/sin lookup tables shaped [maxSeq][ropeHalf].
func makeRoPETables(maxSeq: Int, ropeHalf: Int, theta: Double) -> ([[Float16]], [[Float16]]) {
    let ropeDim = ropeHalf * 2  // = 64 for ZAYA
    var cosT = [[Float16]](repeating: [Float16](repeating: 0, count: ropeHalf), count: maxSeq)
    var sinT = [[Float16]](repeating: [Float16](repeating: 0, count: ropeHalf), count: maxSeq)
    for pos in 0..<maxSeq {
        for i in 0..<ropeHalf {
            let freq  = 1.0 / pow(theta, Double(2 * i) / Double(ropeDim))
            let angle = Double(pos) * freq
            cosT[pos][i] = Float16(cos(angle))
            sinT[pos][i] = Float16(sin(angle))
        }
    }
    return (cosT, sinT)
}

// ---------------------------------------------------------------------------
// ZayaRuntime
// ---------------------------------------------------------------------------

@available(macOS 15.0, *)
final class ZayaRuntime {

    // Loaded once at startup
    let meta: ZayaRuntimeMeta
    let embed: FP16BinaryFile

    // 80 layer models, sorted by layer index
    let layerModels: [MLModel]
    // Cached output name per layer ("attn_out" or "moe_out") — avoid
    // featureNames lookup on hot path. [DB §8.7 peephole micro-optimisation]
    let layerOutputNames: [String]

    // 3 LM head models sorted by vocab_start
    let headModels: [MLModel]
    let headSpecs: [ZayaLMHeadShardSpec]
    let headQueue: DispatchQueue  // concurrent queue for [APL] parallel vocab reduction

    // Dimension shortcuts
    let d: Int

    // ── Persistent buffers (allocated once; mutated in-place per step) ──
    // [EoP §2]: "A computation is correct with respect to a domain of values
    // if it produces correct results for all inputs in that domain."
    // We allocate the maximum required space and reuse it across all tokens.
    let xArr:         MLMultiArray   // (1, d, 1, 1) current hidden state
    let headInputArr: MLMultiArray   // (1, d, 1, 1) stable copy for concurrent head dispatch

    let xPtr:         UnsafeMutablePointer<Float16>  // raw alias into xArr
    let headInputPtr: UnsafeMutablePointer<Float16>  // raw alias into headInputArr

    // Single feature provider for all 80 layer models — holds mutable xArr
    // by reference. Updating xPtr between calls is sufficient; CoreML reads
    // from the live buffer on each prediction(). [HyMT lesson]
    let layerProvider: MLDictionaryFeatureProvider

    // Feature provider for all 3 LM head models
    let headProvider: MLDictionaryFeatureProvider

    // Profile accumulators (reset per runGeneration call)
    var tEmbed  = 0.0
    var tLayers = 0.0
    var tHead   = 0.0
    var profileCalls = 0

    var traceTokens = false

    // ── Stateful attention resources ──────────────────────────────────────
    // Populated only when manifest has "attn_stateful" layers.
    // RoPE tables [MAX_SEQ × ROPE_HALF], attn/kv buffers, MLState per layer.
    let isStateful: Bool
    let ropeHalf:   Int              // ROPE_HALF = 32 for ZAYA
    let maxSeq:     Int              // KV cache length (from manifest max_seq_len)
    let layerSpecs: [ZayaLayerSpec]  // sorted layer specs (parallel to layerModels)
    let ropeCosTable: [[Float16]]    // [maxSeq][ropeHalf] — allocated once
    let ropeSinTable: [[Float16]]    // [maxSeq][ropeHalf]
    let ropeCosArr:      MLMultiArray    // [1, ropeHalf]  — mutated per step
    let ropeSinArr:      MLMultiArray    // [1, ropeHalf]
    let attnMaskArr:     MLMultiArray    // [1, 1, 1, maxSeq]
    let kvWriteMaskArr:  MLMultiArray    // [1, 1, maxSeq, 1]
    let ropeCosPtr:      UnsafeMutablePointer<Float16>
    let ropeSinPtr:      UnsafeMutablePointer<Float16>
    let attnMaskPtr:     UnsafeMutablePointer<Float16>
    let kvWriteMaskPtr:  UnsafeMutablePointer<Float16>
    let attnProvider:    MLDictionaryFeatureProvider?  // nil for probe-only manifests
    var layerStates:     [MLState?]    // one entry per layer (nil for MoE/probe attn)
    var seqPos: Int = 0

    // ── Speculative decode resources (Exp 32) ─────────────────────────────
    // T=vbt verifier: attn_stateful shards have RangeDim lower=1 upper=4.
    // MoE shards: T=1 fixed (legacy) or T=1..4 RangeDim (Exp 34, moe_rangedim=true).
    // With moe_rangedim shards, MoE runs once at T=vbt per verifier pass.
    // See BOOK_ANALYSIS.md Exp 32 for detailed analysis.
    var useSpeculative = false
    var ngramMin       = 1
    var ngramMax       = 8
    var specCalls    = 0
    var specDrafted  = 0
    var specAccepted = 0
    var specFallbacks = 0

    let vbt: Int                         // verifierBatchTokens = meta.rangedimTMax ?? 4
    let verifierXArr: MLMultiArray       // [1, d, vbt, 1]  batch hidden state
    let verifierXPtr: UnsafeMutablePointer<Float16>
    let verifierXStride1: Int            // channel stride = vbt
    let verifierXStride2: Int            // token stride   = 1
    let verifierCosArr: MLMultiArray     // [vbt, ropeHalf]
    let verifierSinArr: MLMultiArray     // [vbt, ropeHalf]
    let verifierCosPtr: UnsafeMutablePointer<Float16>
    let verifierSinPtr: UnsafeMutablePointer<Float16>
    let verifierAttnMaskArr: MLMultiArray     // [1, 1, vbt, maxSeq]
    let verifierKVWriteMaskArr: MLMultiArray  // [1, 1, maxSeq, vbt]
    let verifierAttnMaskPtr: UnsafeMutablePointer<Float16>
    let verifierKVWriteMaskPtr: UnsafeMutablePointer<Float16>
    let verifierAttnMaskStride2: Int     // slot stride  in [1,1,vbt,maxSeq] = maxSeq
    let verifierAttnMaskStride3: Int     // col stride   = 1
    let verifierKVWriteMaskStride2: Int  // row stride   in [1,1,maxSeq,vbt] = vbt
    let verifierKVWriteMaskStride3: Int  // col stride   = 1
    let verifierAttnProvider: MLDictionaryFeatureProvider?  // nil for probe manifests
    let verifierMoeProvider:  MLDictionaryFeatureProvider?  // non-nil when moe_rangedim=true

    init(meta: ZayaRuntimeMeta, embed: FP16BinaryFile,
         layerModels: [MLModel], layerOutputNames: [String], layerSpecs: [ZayaLayerSpec],
         headModels: [MLModel], headSpecs: [ZayaLMHeadShardSpec]) throws {
        self.meta             = meta
        self.embed            = embed
        self.layerModels      = layerModels
        self.layerOutputNames = layerOutputNames
        self.layerSpecs       = layerSpecs
        self.headModels       = headModels
        self.headSpecs        = headSpecs
        self.headQueue        = DispatchQueue(label: "zaya.head", attributes: .concurrent)
        self.d                = meta.dModel

        // [EoP §2] Single allocation per buffer; never reallocated during generation.
        xArr         = try MLMultiArray(shape: [1, d, 1, 1] as [NSNumber], dataType: .float16)
        headInputArr = try MLMultiArray(shape: [1, d, 1, 1] as [NSNumber], dataType: .float16)

        xPtr         = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
        headInputPtr = headInputArr.dataPointer.assumingMemoryBound(to: Float16.self)

        layerProvider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden": MLFeatureValue(multiArray: xArr),
        ])
        headProvider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden": MLFeatureValue(multiArray: headInputArr),
        ])

        // ── Stateful attn resources ───────────────────────────────────────
        let hasStateful = layerSpecs.contains { $0.kind == "attn_stateful" }
        isStateful  = hasStateful
        // ZAYA config: partial_rotary_factor=0.5, d_head=128 → rope_dim=64, rope_half=32
        ropeHalf    = 32
        // KV cache length: from manifest (stateful=2048, probe=131072 but capped)
        maxSeq      = min(meta.maxSeqLen, 8192)
        // Precompute RoPE tables once (rope_theta=5000000)
        (ropeCosTable, ropeSinTable) = makeRoPETables(
            maxSeq: maxSeq, ropeHalf: ropeHalf, theta: 5_000_000.0)

        // Per-step input buffers for stateful attn (T=1 decode, or T up to T_MAX=4)
        ropeCosArr     = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber],    dataType: .float16)
        ropeSinArr     = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber],    dataType: .float16)
        attnMaskArr    = try MLMultiArray(shape: [1, 1, 1, maxSeq] as [NSNumber], dataType: .float16)
        kvWriteMaskArr = try MLMultiArray(shape: [1, 1, maxSeq, 1] as [NSNumber], dataType: .float16)

        ropeCosPtr     = ropeCosArr.dataPointer.assumingMemoryBound(to: Float16.self)
        ropeSinPtr     = ropeSinArr.dataPointer.assumingMemoryBound(to: Float16.self)
        attnMaskPtr    = attnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        kvWriteMaskPtr = kvWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)

        // Initialise attn_mask to all-masked (-10000); we'll unmask 0..seqPos each step.
        let negInf = Float16(-10000)
        for i in 0..<maxSeq { attnMaskPtr[i] = negInf }

        // Provider for attn_stateful layers: input key is "x" (not "hidden")
        if hasStateful {
            attnProvider = try MLDictionaryFeatureProvider(dictionary: [
                "x":            MLFeatureValue(multiArray: xArr),
                "rope_cos":     MLFeatureValue(multiArray: ropeCosArr),
                "rope_sin":     MLFeatureValue(multiArray: ropeSinArr),
                "attn_mask":    MLFeatureValue(multiArray: attnMaskArr),
                "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr),
            ])
        } else {
            attnProvider = nil
        }

        // Create one MLState per attn_stateful layer; nil for MoE/probe-attn layers.
        // [EoP §2] States allocated once; replaced at start of each generation.
        layerStates = layerSpecs.enumerated().map { (i, spec) in
            guard spec.kind == "attn_stateful" else { return nil }
            return layerModels[i].makeState()
        }

        // ── Speculative decode T=vbt verifier buffers (Exp 32) ───────────
        // Allocate once; reused across all verifier passes (zero-alloc hot path).
        vbt = meta.rangedimTMax ?? 4
        verifierXArr = try MLMultiArray(
            shape: [1, NSNumber(value: d), NSNumber(value: vbt), 1], dataType: .float16)
        verifierXPtr     = verifierXArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierXStride1 = Int(truncating: verifierXArr.strides[1])  // channel stride = vbt
        verifierXStride2 = Int(truncating: verifierXArr.strides[2])  // token stride   = 1
        verifierCosArr   = try MLMultiArray(
            shape: [NSNumber(value: vbt), NSNumber(value: ropeHalf)], dataType: .float16)
        verifierSinArr   = try MLMultiArray(
            shape: [NSNumber(value: vbt), NSNumber(value: ropeHalf)], dataType: .float16)
        verifierCosPtr   = verifierCosArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierSinPtr   = verifierSinArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierAttnMaskArr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: vbt), NSNumber(value: maxSeq)], dataType: .float16)
        verifierKVWriteMaskArr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: maxSeq), NSNumber(value: vbt)], dataType: .float16)
        verifierAttnMaskPtr    = verifierAttnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        verifierKVWriteMaskPtr = verifierKVWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        // strides: [1,1,vbt,maxSeq] → stride[2] = maxSeq, stride[3] = 1
        verifierAttnMaskStride2 = Int(truncating: verifierAttnMaskArr.strides[2])
        verifierAttnMaskStride3 = Int(truncating: verifierAttnMaskArr.strides[3])
        // strides: [1,1,maxSeq,vbt] → stride[2] = vbt, stride[3] = 1
        verifierKVWriteMaskStride2 = Int(truncating: verifierKVWriteMaskArr.strides[2])
        verifierKVWriteMaskStride3 = Int(truncating: verifierKVWriteMaskArr.strides[3])
        // Init verifier attn_mask to all-masked (−10000); unmasked selectively per pass.
        let negInfV = Float16(-10000)
        for i in 0..<(vbt * maxSeq) { verifierAttnMaskPtr[i] = negInfV }
        // verifierKVWriteMaskArr default-initialised to zero by MLMultiArray.
        if hasStateful {
            verifierAttnProvider = try MLDictionaryFeatureProvider(dictionary: [
                "x":             MLFeatureValue(multiArray: verifierXArr),
                "rope_cos":      MLFeatureValue(multiArray: verifierCosArr),
                "rope_sin":      MLFeatureValue(multiArray: verifierSinArr),
                "attn_mask":     MLFeatureValue(multiArray: verifierAttnMaskArr),
                "kv_write_mask": MLFeatureValue(multiArray: verifierKVWriteMaskArr),
            ])
        } else {
            verifierAttnProvider = nil
        }
        // Verifier MoE provider: backed by verifierXArr [1,d,vbt,1] — used when
        // moe_rangedim=true so the verifier runs a single T=vbt MoE dispatch
        // instead of t×T=1 sequential dispatches (Exp 34 fix).
        if meta.moeRangedim == true {
            verifierMoeProvider = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: verifierXArr),
            ])
        } else {
            verifierMoeProvider = nil
        }
    }

    // MARK: – Speculative decode helpers

    /// Fill verifier RoPE cos/sin for `slot` at sequence position `pos`.
    @inline(__always)
    private func fillVerifierRoPE(slot: Int, pos: Int) {
        let p = min(pos, maxSeq - 1)
        let rh = ropeHalf
        for i in 0..<rh {
            verifierCosPtr[slot * rh + i] = ropeCosTable[p][i]
            verifierSinPtr[slot * rh + i] = ropeSinTable[p][i]
        }
    }

    /// Reset verifier attn_mask to all-negInf and kv_write_mask to all-zero.
    /// Called at the start of every forwardVerifier pass. [EoP §2]
    private func resetVerifierInputs() {
        let negInf = Float16(-10000)
        let total  = vbt * maxSeq
        for i in 0..<total { verifierAttnMaskPtr[i]    = negInf       }
        for i in 0..<total { verifierKVWriteMaskPtr[i] = Float16(0.0) }
    }

    /// Run `tokens.count` (1…vbt) tokens through all 80 layers:
    ///   • attn_stateful layers → single T=vbt ANE call (RangeDim).
    ///   • MoE layers           → t × T=1 ANE calls (MoE shards are T=1 fixed).
    /// After the layer chain, returns argmax predictions for slots 0..<tokens.count,
    /// or [] when advanceOnly=true (used for non-final prefill chunks).
    ///
    /// posStart:    KV-cache position of tokens[0] in this pass.
    /// cacheSeqLen: already-committed entries before this pass.
    func forwardVerifier(tokens: [Int], posStart: Int, cacheSeqLen: Int,
                         advanceOnly: Bool = false) throws -> [Int] {
        precondition(!tokens.isEmpty && tokens.count <= vbt)
        guard let vap = verifierAttnProvider else { return [] }
        let t = tokens.count

        // 1. Reset masks; fill RoPE for all vbt slots (unused slots are safe —
        //    their attn_mask rows stay negInf and kv_write_mask cols stay 0).
        resetVerifierInputs()
        for slot in 0..<vbt {
            fillVerifierRoPE(slot: slot, pos: posStart + slot)
        }

        // 2. Embed each draft token; build causal attn_mask + kv_write_mask.
        for slot in 0..<t {
            embed.writeRow(tokens[slot], dim: d, into: verifierXPtr,
                           channelStride: verifierXStride1, tokenSlot: slot)
            let visEnd = min(maxSeq - 1, cacheSeqLen + slot)
            if visEnd >= 0 {
                for j in 0...visEnd {
                    verifierAttnMaskPtr[slot * verifierAttnMaskStride2
                                        + j   * verifierAttnMaskStride3] = Float16(0.0)
                }
            }
            verifierKVWriteMaskPtr[(cacheSeqLen + slot) * verifierKVWriteMaskStride2
                                   + slot * verifierKVWriteMaskStride3] = Float16(1.0)
        }

        // 3. Layer chain: T=vbt for attn, t×T=1 for MoE.
        //    [Exp 32] Measured attn speedup ~4× (ANE parallel); MoE runs t times
        //    sequentially — net ~12% decode speedup (MoE-dominated architecture).
        for idx in 0..<layerModels.count {
            let spec    = layerSpecs[idx]
            let outName = layerOutputNames[idx]
            if spec.kind == "attn_stateful", let state = layerStates[idx] {
                // T=vbt batch attn — one ANE dispatch for all vbt token positions.
                try autoreleasepool {
                    let result = try layerModels[idx].prediction(from: vap, using: state)
                    let output = result.featureValue(for: outName)!.multiArrayValue!
                    copyHiddenTokens(output, into: verifierXPtr, d: d,
                                     batchTokens: vbt,
                                     dstChannelStride: verifierXStride1,
                                     dstTokenStride:   verifierXStride2)
                }
            } else if let vmp = verifierMoeProvider {
                // MoE T=vbt — single ANE dispatch for all vbt token positions.
                // [Exp 34] With RangeDim MoE shards, one dispatch replaces t×T=1 serial
                // calls → verifier MoE cost drops from t×110ms to ~110ms fixed.
                try autoreleasepool {
                    let result = try layerModels[idx].prediction(from: vmp)
                    let output = result.featureValue(for: outName)!.multiArrayValue!
                    copyHiddenTokens(output, into: verifierXPtr, d: d,
                                     batchTokens: vbt,
                                     dstChannelStride: verifierXStride1,
                                     dstTokenStride:   verifierXStride2)
                }
            } else {
                // MoE layer — T=1 fixed; run once per draft token slot.
                // Column extract → MoE dispatch → column insert.
                for slot in 0..<t {
                    // Extract verifierXArr[:,slot] → xArr
                    for ch in 0..<d {
                        xPtr[ch] = verifierXPtr[ch * verifierXStride1 + slot * verifierXStride2]
                    }
                    try autoreleasepool {
                        let result = try layerModels[idx].prediction(from: layerProvider)
                        let output = result.featureValue(for: outName)!.multiArrayValue!
                        copyFlatFloat16(output, into: xPtr, count: d)
                    }
                    // Insert xArr back → verifierXArr[:,slot]
                    for ch in 0..<d {
                        verifierXPtr[ch * verifierXStride1 + slot * verifierXStride2] = xPtr[ch]
                    }
                }
            }
        }

        if advanceOnly { return [] }
        return try predictSlotsWithT1Head(count: t)
    }

    /// Run the 3-shard T=1 LM head for each slot 0..<count and return argmax.
    /// Fills headInputArr from verifierXArr one column at a time. [APL parallel]
    private func predictSlotsWithT1Head(count: Int) throws -> [Int] {
        var results = [Int](repeating: 0, count: count)
        for slot in 0..<count {
            for ch in 0..<d {
                headInputPtr[ch] = verifierXPtr[ch * verifierXStride1 + slot * verifierXStride2]
            }
            let group      = DispatchGroup()
            var shardLogits = [[Float]](repeating: [], count: headModels.count)
            let lock        = NSLock()
            var shardError: Error? = nil
            for s in 0..<headModels.count {
                group.enter()
                headQueue.async {
                    do {
                        let res = try self.headModels[s].prediction(from: self.headProvider)
                        let arr = res.featureValue(for: "logits")!.multiArrayValue!
                        let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                        let st  = arr.strides.map { Int(truncating: $0) }
                        let s1  = st.count > 1 ? st[1] : 1
                        let spec = self.headSpecs[s]
                        let n   = spec.vocabEnd - spec.vocabStart
                        var f   = [Float](repeating: 0, count: n)
                        for i in 0..<n { f[i] = Float(ptr[i * s1]) }
                        lock.lock(); shardLogits[s] = f; lock.unlock()
                    } catch {
                        lock.lock()
                        if shardError == nil { shardError = error }
                        lock.unlock()
                    }
                    group.leave()
                }
            }
            group.wait()
            if let err = shardError { throw err }
            var all = [Float](); all.reserveCapacity(meta.vocabSize)
            for s in 0..<headModels.count { all.append(contentsOf: shardLogits[s]) }
            // Greedy argmax — verifier always compares against target argmax.
            results[slot] = sampleLogits(all, temperature: 1.0, topK: 1, topP: 1.0)
        }
        return results
    }

    /// Build a draft sequence for speculative decode.
    /// Returns [firstToken] + up to (vbt-1) n-gram continuations.
    func speculativeDraft(history: [Int], firstToken: Int) -> [Int] {
        var draft   = [firstToken]
        var scratch = history
        scratch.append(firstToken)
        while draft.count < vbt {
            guard let p = findNGramProposal(history: scratch,
                                            minN: ngramMin, maxN: ngramMax) else { break }
            draft.append(p.token)
            scratch.append(p.token)
        }
        return draft
    }

    // MARK: – Forward one token

    /// Forward a single token through all 80 layers + LM head → next token id.
    /// skipHead = true during prefill: [DB §8.7] avoid 3 concurrent head invocations
    /// whose logits would be discarded.
    @discardableResult
    func forwardOne(tokenId: Int,
                    collectProfile: Bool = false,
                    skipHead: Bool = false,
                    temperature: Float = 1.0,
                    topK: Int = 0,
                    topP: Float = 1.0,
                    repPenalty: Float = 1.0,
                    seen: [Int] = []) throws -> Int {

        // 1. Embedding lookup — O(d) memcpy, no compute. [EoP §2]
        let t0 = CFAbsoluteTimeGetCurrent()
        embed.writeRow(tokenId, dim: d, into: xPtr)
        let t1 = CFAbsoluteTimeGetCurrent()

        // 2. 80 layer passes on ANE — all compute on Neural Engine.
        // For stateful attn layers: fill RoPE/mask buffers once per step, then
        // call prediction(from:using:) with the per-layer MLState.
        // For probe-attn or MoE layers: use the shared layerProvider (no state).

        if isStateful {
            // ── Fill RoPE buffers for position seqPos ─────────────────────
            // [T=1 decode only; extend for prefill chunking if needed]
            for i in 0..<ropeHalf {
                ropeCosPtr[i] = ropeCosTable[min(seqPos, maxSeq - 1)][i]
                ropeSinPtr[i] = ropeSinTable[min(seqPos, maxSeq - 1)][i]
            }
            // ── Fill attn_mask: unmask positions 0..seqPos ────────────────
            // attnMaskArr was initialised to all-negInf at startup.
            // Each step we set one more position to 0.0.
            let pos = min(seqPos, maxSeq - 1)
            attnMaskPtr[pos] = Float16(0)

            // ── Fill kv_write_mask: write only at seqPos ──────────────────
            // Previous step's 1.0 is already overwritten by the clear below.
            let prevPos = seqPos == 0 ? 0 : seqPos - 1  // clear prev write bit
            kvWriteMaskPtr[prevPos] = Float16(0)         // safe (0×anything = 0)
            kvWriteMaskPtr[pos]     = Float16(1)
        }

        for idx in 0..<layerModels.count {
            let spec   = layerSpecs[idx]
            let result: MLFeatureProvider

            if spec.kind == "attn_stateful", let ap = attnProvider,
               let state = layerStates[idx] {
                // Stateful GQA attention: 5 inputs + MLState KV cache
                result = try layerModels[idx].prediction(from: ap, using: state)
            } else {
                // Probe attn ("attn_out") or MoE ("moe_out"): simple hidden→output
                result = try layerModels[idx].prediction(from: layerProvider)
            }
            let output = result.featureValue(for: layerOutputNames[idx])!.multiArrayValue!
            copyFlatFloat16(output, into: xPtr, count: d)
        }

        if isStateful { seqPos = min(seqPos + 1, maxSeq - 1) }
        let t2 = CFAbsoluteTimeGetCurrent()

        if skipHead {
            if collectProfile {
                tEmbed  += t1 - t0
                tLayers += t2 - t1
                profileCalls += 1
            }
            return -1
        }

        // 3. Copy hidden → headInputArr for safe concurrent read during head dispatch.
        // xArr will not be mutated again until after group.wait(). [EoP §2]
        memcpy(headInputPtr, xPtr, d * MemoryLayout<Float16>.size)

        // 4. [APL] Parallel vocab-partition reduction: 3 head shards dispatched
        // concurrently, each computing logits over ~87k vocab entries independently.
        // This is Iverson's inner product decomposed across disjoint output ranges.
        let group = DispatchGroup()
        var shardLogits = [[Float]](repeating: [], count: headModels.count)
        let lock  = NSLock()
        var shardError: Error? = nil

        for s in 0..<headModels.count {
            group.enter()
            headQueue.async {
                do {
                    let res = try self.headModels[s].prediction(from: self.headProvider)
                    let arr = res.featureValue(for: "logits")!.multiArrayValue!
                    let ptr = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                    let strides = arr.strides.map { Int(truncating: $0) }
                    let stride1 = strides.count > 1 ? strides[1] : 1
                    let spec = self.headSpecs[s]
                    let n = spec.vocabEnd - spec.vocabStart
                    var floats = [Float](repeating: 0, count: n)
                    for i in 0..<n { floats[i] = Float(ptr[i * stride1]) }
                    lock.lock(); shardLogits[s] = floats; lock.unlock()
                } catch {
                    lock.lock()
                    if shardError == nil { shardError = error }
                    lock.unlock()
                }
                group.leave()
            }
        }
        group.wait()
        if let err = shardError { throw err }
        let t3 = CFAbsoluteTimeGetCurrent()

        // 5. Concatenate shard logits and sample
        var allLogits = [Float]()
        allLogits.reserveCapacity(meta.vocabSize)
        for s in 0..<headModels.count { allLogits.append(contentsOf: shardLogits[s]) }
        let next = sampleLogits(allLogits, temperature: temperature,
                                topK: topK, topP: topP,
                                penalty: repPenalty, seen: seen)

        if collectProfile {
            tEmbed  += t1 - t0
            tLayers += t2 - t1
            tHead   += t3 - t2
            profileCalls += 1
        }
        if traceTokens { print("tok[\(tokenId)]→\(next)") }
        return next
    }

    // MARK: – Autoregressive generation (speculative)

    /// Speculative decode: T=vbt chunked prefill + n-gram draft + T=vbt verifier.
    /// Architecture note: attn T=vbt (ANE parallel); MoE t×T=1 sequential.
    /// Net decode speedup ≈ 12% (attn=15% of compute; MoE=85%). [Exp 32]
    func runGenerationSpeculative(promptIds: [Int], maxNew: Int,
                                  requestProfile: Bool,
                                  temperature: Float = 1.0,
                                  topK: Int = 0,
                                  topP: Float = 1.0,
                                  repPenalty: Float = 1.0)
        throws -> (generated: [Int], timing: ZayaServeTiming, profile: ZayaServeProfile?)
    {
        precondition(!promptIds.isEmpty && isStateful)
        specCalls = 0; specDrafted = 0; specAccepted = 0; specFallbacks = 0
        tEmbed = 0; tLayers = 0; tHead = 0; profileCalls = 0

        // Fresh KV cache for this generation.
        seqPos = 0
        let negInf = Float16(-10000)
        for i in 0..<maxSeq { attnMaskPtr[i] = negInf }
        for i in 0..<maxSeq { kvWriteMaskPtr[i] = Float16(0) }
        layerStates = layerSpecs.enumerated().map { (i, spec) in
            guard spec.kind == "attn_stateful" else { return nil }
            return layerModels[i].makeState()
        }

        var generated   = [Int]()
        var cacheSeqLen = 0

        // ── Chunked prefill: T=vbt blocks ─────────────────────────────────
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var pi = 0
        var next: Int = meta.bosTokenId
        while pi < promptIds.count {
            let chunkEnd = min(pi + vbt, promptIds.count)
            let chunk    = Array(promptIds[pi..<chunkEnd])
            let isLast   = (chunkEnd == promptIds.count)
            if isLast {
                let preds = try forwardVerifier(tokens: chunk,
                                                posStart: cacheSeqLen,
                                                cacheSeqLen: cacheSeqLen)
                next = preds[chunk.count - 1]
            } else {
                _ = try forwardVerifier(tokens: chunk,
                                        posStart: cacheSeqLen,
                                        cacheSeqLen: cacheSeqLen,
                                        advanceOnly: true)
            }
            cacheSeqLen += chunk.count
            pi += chunk.count
        }
        let prefillElapsed = CFAbsoluteTimeGetCurrent() - prefillStart
        generated.append(next)

        // Sync T=1 attn_mask so forwardOne fallback calls see correct context.
        for j in 0..<cacheSeqLen { attnMaskPtr[j] = Float16(0.0) }
        seqPos = cacheSeqLen

        // ── Speculative decode loop ────────────────────────────────────────
        let decodeStart = CFAbsoluteTimeGetCurrent()
        while generated.count < maxNew {
            if generated.last == meta.eosTokenId { break }
            let history = promptIds + generated
            let draft   = speculativeDraft(history: history, firstToken: generated.last!)
            // posStart = current committed length (1 target anchor + 0..vbt-1 drafts)
            let preds = try forwardVerifier(tokens: draft,
                                            posStart: cacheSeqLen,
                                            cacheSeqLen: cacheSeqLen)
            specCalls   += 1
            specDrafted += max(0, draft.count - 1)

            var acceptedCount  = 0
            var emittedFallback = false
            if draft.count > 1 {
                for idx in 1..<draft.count {
                    if preds[idx - 1] == draft[idx] {
                        generated.append(draft[idx])
                        acceptedCount += 1; specAccepted += 1
                        if generated.count >= maxNew { break }
                    } else {
                        generated.append(preds[idx - 1])
                        emittedFallback = true; specFallbacks += 1
                        break
                    }
                }
            }
            if !emittedFallback && generated.count < maxNew {
                generated.append(preds[draft.count - 1])
            }
            cacheSeqLen += 1 + acceptedCount
            // Keep T=1 attn_mask in sync for the next prefill/decode cycle.
            for j in seqPos..<min(cacheSeqLen, maxSeq) { attnMaskPtr[j] = Float16(0.0) }
            seqPos = cacheSeqLen
        }
        let decodeElapsed  = CFAbsoluteTimeGetCurrent() - decodeStart
        let decodeTokens   = max(0, generated.count - 1)
        let totalSec       = prefillElapsed + decodeElapsed
        let totalForward   = (specCalls > 0 ? specCalls : 0) + (promptIds.count + vbt - 1) / vbt
        let timing = ZayaServeTiming(
            prefillTokens:          promptIds.count,
            prefillSeconds:         prefillElapsed,
            decodeTokens:           decodeTokens,
            decodeSeconds:          decodeElapsed,
            decodeTokensPerSecond:  decodeElapsed > 0 && decodeTokens > 0
                                        ? Double(decodeTokens) / decodeElapsed : 0,
            forwardCalls:           totalForward,
            forwardSeconds:         totalSec,
            forwardTokensPerSecond: totalSec > 0 ? Double(totalForward) / totalSec : 0
        )
        return (generated, timing, nil)
    }

    // MARK: – Autoregressive generation

    func runGeneration(promptIds: [Int], maxNew: Int,
                       requestProfile: Bool,
                       temperature: Float = 1.0,
                       topK: Int = 0,
                       topP: Float = 1.0,
                       repPenalty: Float = 1.0)
        throws -> (generated: [Int], timing: ZayaServeTiming, profile: ZayaServeProfile?)
    {
        // Route to speculative path when requested and shards support it.
        if useSpeculative && isStateful {
            return try runGenerationSpeculative(
                promptIds: promptIds, maxNew: maxNew, requestProfile: requestProfile,
                temperature: temperature, topK: topK, topP: topP, repPenalty: repPenalty)
        }

        precondition(!promptIds.isEmpty, "prompt must not be empty")
        tEmbed = 0; tLayers = 0; tHead = 0; profileCalls = 0

        // ── Reset stateful KV cache for each new generation ───────────────
        // [EoP §2] Fresh MLState objects (zeroed) replace old ones so KV from
        // previous requests doesn't bleed into the new context.
        if isStateful {
            seqPos = 0
            // Reset attn_mask back to all-masked
            let negInf = Float16(-10000)
            for i in 0..<maxSeq { attnMaskPtr[i] = negInf }
            // Reset kv_write_mask
            for i in 0..<maxSeq { kvWriteMaskPtr[i] = Float16(0) }
            // Recreate MLState per stateful layer (fast — just zeroed allocations)
            layerStates = layerSpecs.enumerated().map { (i, spec) in
                guard spec.kind == "attn_stateful" else { return nil }
                return layerModels[i].makeState()
            }
        }

        let prefillStart = CFAbsoluteTimeGetCurrent()

        // [DB §8.7] Prefill: skip LM head for all prompt tokens except the last.
        if promptIds.count > 1 {
            for tokenId in promptIds.dropLast() {
                _ = try forwardOne(tokenId: tokenId,
                                   collectProfile: requestProfile,
                                   skipHead: true)
            }
        }

        let prefillEnd = CFAbsoluteTimeGetCurrent()

        // Decode: forward last prompt token → first generated token.
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var seenTokens = promptIds
        var generated  = [Int]()
        var nextToken  = try forwardOne(tokenId: promptIds.last!,
                                        collectProfile: requestProfile,
                                        temperature: temperature,
                                        topK: topK, topP: topP,
                                        repPenalty: repPenalty,
                                        seen: seenTokens)
        generated.append(nextToken)
        seenTokens.append(nextToken)

        let forwardCalls = promptIds.count + generated.count

        while generated.count < maxNew && nextToken != meta.eosTokenId {
            nextToken = try forwardOne(tokenId: nextToken,
                                       collectProfile: requestProfile,
                                       temperature: temperature,
                                       topK: topK, topP: topP,
                                       repPenalty: repPenalty,
                                       seen: seenTokens)
            generated.append(nextToken)
            seenTokens.append(nextToken)
        }
        let decodeEnd = CFAbsoluteTimeGetCurrent()

        let prefillSec = prefillEnd  - prefillStart
        let decodeSec  = decodeEnd   - decodeStart
        let totalSec   = prefillSec  + decodeSec
        let totalCalls = promptIds.count + generated.count

        let timing = ZayaServeTiming(
            prefillTokens:         promptIds.count,
            prefillSeconds:        prefillSec,
            decodeTokens:          generated.count,
            decodeSeconds:         decodeSec,
            decodeTokensPerSecond: generated.count > 0 ? Double(generated.count) / decodeSec : 0,
            forwardCalls:          forwardCalls,
            forwardSeconds:        totalSec,
            forwardTokensPerSecond: totalCalls > 0 ? Double(totalCalls) / totalSec : 0
        )

        var prof: ZayaServeProfile? = nil
        if requestProfile && profileCalls > 0 {
            prof = ZayaServeProfile(
                calls:        profileCalls,
                embedSeconds: tEmbed,
                layersSeconds: tLayers,
                headSeconds:  tHead
            )
        }
        return (generated, timing, prof)
    }
}

// ---------------------------------------------------------------------------
// Unix socket daemon helpers (reused pattern from HyMT)
// ---------------------------------------------------------------------------

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
func runSocketServer(socketPath: String, runtime: ZayaRuntime, warmupCalls: Int) throws {
    let serverFd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard serverFd >= 0 else {
        throw NSError(domain: "ZayaSocket", code: Int(errno),
                      userInfo: [NSLocalizedDescriptionKey: "socket() failed"])
    }
    var reuse: Int32 = 1
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
    unlink(socketPath)

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    guard socketPath.utf8.count < MemoryLayout.size(ofValue: addr.sun_path) else {
        throw NSError(domain: "ZayaSocket", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Socket path too long"])
    }
    withUnsafeMutableBytes(of: &addr.sun_path) { buf in
        socketPath.withCString { src in _ = strcpy(buf.baseAddress!.assumingMemoryBound(to: CChar.self), src) }
    }
    let bindResult = withUnsafePointer(to: addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.bind(serverFd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard bindResult == 0 else {
        Darwin.close(serverFd)
        throw NSError(domain: "ZayaSocket", code: Int(errno),
                      userInfo: [NSLocalizedDescriptionKey: "bind() failed: \(strerror(errno))"])
    }
    Darwin.listen(serverFd, 8)

    print("READY {\"protocol\":\"zaya-jsonl-v1\",\"transport\":\"unix-socket\",\"path\":\"\(socketPath)\"}")
    fflush(stdout)
    printStderr("Listening on \(socketPath)")

    while true {
        let clientFd = Darwin.accept(serverFd, nil, nil)
        guard clientFd >= 0 else { continue }
        printStderr("Client connected")

        var reader = FDLineReader(fd: clientFd)
        while let line = reader.readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            do {
                let req = try JSONDecoder().decode(ZayaServeRequest.self, from: Data(trimmed.utf8))
                let (generated, timing, prof) = try runtime.runGeneration(
                    promptIds:    req.promptIds,
                    maxNew:       req.maxNew ?? 50,
                    requestProfile: req.profile ?? false,
                    temperature:  req.temperature ?? 1.0,
                    topK:         req.topK ?? 0,
                    topP:         req.topP ?? 1.0,
                    repPenalty:   req.repPenalty ?? 1.0)
                let resp = ZayaServeResponse(ok: true, generatedIds: generated,
                                             timing: timing, profile: prof, error: nil)
                writeLineFD(clientFd, try encodeJSON(resp))
            } catch {
                let resp = ZayaServeResponse(ok: false, generatedIds: nil,
                                             timing: nil, profile: nil,
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
    var metaPath      = "tmp/zaya_shards/zaya_runtime_meta.json"
    var promptIds     = [Int]()
    var maxNew        = 20
    var warmupCalls   = 0
    var profile       = false
    var serve         = false
    var traceTokens   = false
    var unixSocketPath: String? = nil
    var temperature: Float = 1.0
    var topK:  Int   = 0
    var topP:  Float = 1.0
    var repPenalty: Float = 1.0
    var speculative = false
    var ngramMin    = 1
    var ngramMax    = 8

    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--meta":         metaPath       = args[i+1]; i += 2
        case "--prompt-ids":   promptIds      = args[i+1].split(separator: ",")
                               .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }; i += 2
        case "--max-new":      maxNew         = Int(args[i+1])!; i += 2
        case "--warmup":       warmupCalls    = Int(args[i+1])!; i += 2
        case "--temperature":  temperature    = Float(args[i+1])!; i += 2
        case "--top-k":        topK           = Int(args[i+1])!; i += 2
        case "--top-p":        topP           = Float(args[i+1])!; i += 2
        case "--rep-pen":      repPenalty     = Float(args[i+1])!; i += 2
        case "--profile":      profile        = true; i += 1
        case "--serve":        serve          = true; i += 1
        case "--trace":        traceTokens    = true; i += 1
        case "--unix-socket":  unixSocketPath = args[i+1]; i += 2
        case "--speculative":  speculative    = true; i += 1
        case "--ngram-min":    ngramMin       = Int(args[i+1])!; i += 2
        case "--ngram-max":    ngramMax       = Int(args[i+1])!; i += 2
        default:               i += 1
        }
    }

    func status(_ msg: String) {
        if serve { printStderr(msg) } else { print(msg) }
    }

    // ── Load meta ────────────────────────────────────────────────────────
    let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    let meta     = try JSONDecoder().decode(ZayaRuntimeMeta.self, from: metaData)
    let sorted   = meta.layers.sorted { $0.layer < $1.layer }
    precondition(sorted.count == meta.nLayers,
        "layers count \(sorted.count) ≠ n_layers \(meta.nLayers)")

    status("Loading \(meta.modelFamily): \(meta.nLayers)L d=\(meta.dModel) vocab=\(meta.vocabSize) experts=\(meta.nExperts)")

    // ── Load embedding binary ────────────────────────────────────────────
    let embedPath = resolvePath(meta.embedBin, relativeTo: metaPath)
    let embed = try FP16BinaryFile(path: embedPath, expectedCount: meta.vocabSize * meta.dModel)
    status(String(format: "Embed: %.2f GB memory-mapped", Double(embed.count * 2) / 1e9))

    guard #available(macOS 15.0, *) else {
        fputs("ERROR: requires macOS 15+\n", stderr); exit(1)
    }

    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine

    // ── Load 80 layer models ─────────────────────────────────────────────
    status("Loading \(sorted.count) layer shards...")
    var layerModels      = [MLModel]()
    var layerOutputNames = [String]()
    layerModels.reserveCapacity(sorted.count)
    layerOutputNames.reserveCapacity(sorted.count)

    for spec in sorted {
        let path     = resolvePath(spec.mlmodelc, relativeTo: metaPath)
        let outName: String
        switch spec.kind {
        case "attn":          outName = "attn_out"   // probe shard
        case "attn_stateful": outName = "hidden"     // stateful GQA shard
        default:              outName = "moe_out"    // MoE shard
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let m  = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        status(String(format: "  L%02d (%@) %.2fs", spec.layer, spec.kind, dt))
        layerModels.append(m)
        layerOutputNames.append(outName)
    }

    // ── Load 3 LM head models ────────────────────────────────────────────
    let sortedHead = meta.lmHeadShards.sorted { $0.vocabStart < $1.vocabStart }
    status("Loading \(sortedHead.count) LM head shards...")
    var headModels = [MLModel]()
    headModels.reserveCapacity(sortedHead.count)
    for spec in sortedHead {
        let path = resolvePath(spec.mlmodelc, relativeTo: metaPath)
        let t0   = CFAbsoluteTimeGetCurrent()
        let m    = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
        let dt   = CFAbsoluteTimeGetCurrent() - t0
        status(String(format: "  vocab[%d,%d) %.2fs", spec.vocabStart, spec.vocabEnd, dt))
        headModels.append(m)
    }

    // ── Build runtime ────────────────────────────────────────────────────
    let runtime = try ZayaRuntime(
        meta: meta, embed: embed,
        layerModels: layerModels, layerOutputNames: layerOutputNames, layerSpecs: sorted,
        headModels: headModels, headSpecs: sortedHead)
    runtime.traceTokens    = traceTokens
    runtime.useSpeculative = speculative
    runtime.ngramMin       = ngramMin
    runtime.ngramMax       = ngramMax
    status("Ready. BOS=\(meta.bosTokenId) EOS=\(meta.eosTokenId)")

    // ── Warmup ───────────────────────────────────────────────────────────
    if warmupCalls > 0 {
        status("Warming up (\(warmupCalls) call(s))...")
        for _ in 0..<warmupCalls {
            _ = try runtime.runGeneration(promptIds: [meta.bosTokenId],
                                          maxNew: 1, requestProfile: false)
        }
        status("Warmup done.")
    }

    // ── Dispatch mode ────────────────────────────────────────────────────
    if let sockPath = unixSocketPath {
        try runSocketServer(socketPath: sockPath, runtime: runtime, warmupCalls: warmupCalls)

    } else if serve {
        // Stdin/stdout JSON-line serve mode
        print("READY {\"protocol\":\"zaya-jsonl-v1\"}")
        fflush(stdout)
        while let line = readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            do {
                let req = try JSONDecoder().decode(ZayaServeRequest.self, from: Data(trimmed.utf8))
                let (generated, timing, prof) = try runtime.runGeneration(
                    promptIds:    req.promptIds,
                    maxNew:       req.maxNew ?? 50,
                    requestProfile: req.profile ?? false,
                    temperature:  req.temperature ?? 1.0,
                    topK:         req.topK ?? 0,
                    topP:         req.topP ?? 1.0,
                    repPenalty:   req.repPenalty ?? 1.0)
                let resp = ZayaServeResponse(ok: true, generatedIds: generated,
                                             timing: timing, profile: prof, error: nil)
                print(try encodeJSON(resp))
            } catch {
                let resp = ZayaServeResponse(ok: false, generatedIds: nil,
                                             timing: nil, profile: nil,
                                             error: error.localizedDescription)
                print((try? encodeJSON(resp)) ?? "{\"ok\":false}")
            }
            fflush(stdout)
        }

    } else {
        // Single-shot mode
        if promptIds.isEmpty { promptIds = [meta.bosTokenId] }
        let (generated, timing, prof) = try runtime.runGeneration(
            promptIds: promptIds, maxNew: maxNew, requestProfile: profile,
            temperature: temperature, topK: topK, topP: topP, repPenalty: repPenalty)
        print("Generated IDs: \(generated)")
        print(String(format: "Prefill: %d tok in %.3fs",
                     timing.prefillTokens, timing.prefillSeconds))
        print(String(format: "Decode:  %d tok in %.3fs → %.2f tok/s",
                     timing.decodeTokens, timing.decodeSeconds, timing.decodeTokensPerSecond))
        print(String(format: "Total:   %d fwd in %.3fs → %.2f tok/s",
                     timing.forwardCalls, timing.forwardSeconds, timing.forwardTokensPerSecond))
        if let p = prof {
            print(String(format: "Profile: calls=%d embed=%.3fs layers=%.3fs head=%.3fs",
                         p.calls, p.embedSeconds, p.layersSeconds, p.headSeconds))
        }
        if speculative && runtime.specCalls > 0 {
            let draftedTotal = runtime.specDrafted
            let rate = draftedTotal > 0
                ? Double(runtime.specAccepted) / Double(draftedTotal) : 0
            print(String(format: "Spec:    verifier_calls=%d drafted=%d accepted=%d fallbacks=%d  acceptance=%.1f%%",
                         runtime.specCalls, runtime.specDrafted,
                         runtime.specAccepted, runtime.specFallbacks, rate * 100))
        }
    }
}

try main()
