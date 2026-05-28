// Phi-4-mini ANE smoke runtime.
//
// Heavy compute path is ANE-only: 32 stateful transformer layer shards plus
// 4 final RMSNorm+tied-LM-head shards. Host work is limited to token-id
// embedding lookup, RoPE/mask bookkeeping, and argmax sampling.

import CoreML
import Foundation

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

struct SpeculativeVerifierSpec: Decodable {
    let batchTokens: Int
    let layers: [LayerSpec]
    let lmHeadShards: [LMHeadShardSpec]

    enum CodingKeys: String, CodingKey {
        case batchTokens = "batch_tokens"
        case layers
        case lmHeadShards = "lm_head_shards"
    }
}

struct PhiRuntimeMeta: Decodable {
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
    let speculativeVerifier: SpeculativeVerifierSpec?

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
        case speculativeVerifier = "speculative_verifier"
    }
}

struct PhiServeRequest: Decodable {
    let promptIds: [Int]
    let maxNew: Int?
    let profile: Bool?
    let structuredCoT: Bool?

    enum CodingKeys: String, CodingKey {
        case promptIds = "prompt_ids"
        case maxNew = "max_new"
        case profile
        case structuredCoT = "structured_cot"
    }
}

struct PhiServeTiming: Encodable {
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

struct PhiServeProfile: Encodable {
    let calls: Int
    let embedSeconds: Double
    let ropeMaskSeconds: Double
    let layersSeconds: Double
    let headCopySeconds: Double
    let headPredictReduceSeconds: Double
    let headPredictShardWorkSeconds: Double
    let headReduceShardWorkSeconds: Double

    enum CodingKeys: String, CodingKey {
        case calls
        case embedSeconds = "embed_s"
        case ropeMaskSeconds = "rope_mask_s"
        case layersSeconds = "layers_s"
        case headCopySeconds = "head_copy_s"
        case headPredictReduceSeconds = "head_predict_reduce_s"
        case headPredictShardWorkSeconds = "head_predict_shard_work_s"
        case headReduceShardWorkSeconds = "head_reduce_shard_work_s"
    }
}

struct PhiStructuredCoTStats: Encodable {
    let name: String
    let forcedTokens: Int
    let fieldContentTokens: Int
    let fieldNewlineTokens: Int
    let openTokens: Int
    let fieldsCompleted: Int
    let activeStage: String
    let fieldTokenCounts: [String: Int]

    enum CodingKeys: String, CodingKey {
        case name
        case forcedTokens = "forced_tokens"
        case fieldContentTokens = "field_content_tokens"
        case fieldNewlineTokens = "field_newline_tokens"
        case openTokens = "open_tokens"
        case fieldsCompleted = "fields_completed"
        case activeStage = "active_stage"
        case fieldTokenCounts = "field_token_counts"
    }
}

struct StructuredCoTStageSpec: Decodable {
    let kind: String
    let name: String
    let tokenIds: [Int]?
    let minTokens: Int?
    let maxTokens: Int?

    enum CodingKeys: String, CodingKey {
        case kind
        case name
        case tokenIds = "token_ids"
        case minTokens = "min_tokens"
        case maxTokens = "max_tokens"
    }
}

struct StructuredCoTManifest: Decodable {
    let name: String
    let newlineTokenId: Int
    let stopTokenIds: [Int]
    let stages: [StructuredCoTStageSpec]

    enum CodingKeys: String, CodingKey {
        case name
        case newlineTokenId = "newline_token_id"
        case stopTokenIds = "stop_token_ids"
        case stages
    }
}

struct TokenConstraint {
    let forcedToken: Int?
    let blockedTokenIds: Set<Int>
    let reason: String

    init(forcedToken: Int, reason: String) {
        self.forcedToken = forcedToken
        self.blockedTokenIds = []
        self.reason = reason
    }

    init(blockedTokenIds: Set<Int>, reason: String) {
        self.forcedToken = nil
        self.blockedTokenIds = blockedTokenIds
        self.reason = reason
    }

    func allows(_ tokenId: Int) -> Bool {
        if let forcedToken { return tokenId == forcedToken }
        return !blockedTokenIds.contains(tokenId)
    }
}

final class StructuredCoTSampler {
    let manifest: StructuredCoTManifest
    private let stopTokenIds: Set<Int>
    private var stageIndex = 0
    private var literalOffset = 0
    private var fieldContentCount = 0
    private var forcedTokens = 0
    private var fieldContentTokens = 0
    private var fieldNewlineTokens = 0
    private var openTokens = 0
    private var fieldsCompleted = 0
    private var fieldTokenCounts = [String: Int]()

    init(manifest: StructuredCoTManifest) {
        self.manifest = manifest
        self.stopTokenIds = Set(manifest.stopTokenIds)
    }

    func constraintForNextToken() -> TokenConstraint? {
        while stageIndex < manifest.stages.count {
            let stage = manifest.stages[stageIndex]
            switch stage.kind {
            case "literal":
                let ids = stage.tokenIds ?? []
                if literalOffset < ids.count {
                    return TokenConstraint(forcedToken: ids[literalOffset], reason: stage.name)
                }
                stageIndex += 1
                literalOffset = 0
            case "field":
                let minTokens = stage.minTokens ?? 0
                let maxTokens = stage.maxTokens ?? 24
                if fieldContentCount >= maxTokens {
                    return TokenConstraint(forcedToken: manifest.newlineTokenId, reason: "\(stage.name).newline")
                }
                var blocked = stopTokenIds
                if fieldContentCount < minTokens { blocked.insert(manifest.newlineTokenId) }
                return TokenConstraint(blockedTokenIds: blocked, reason: stage.name)
            case "open":
                return nil
            default:
                stageIndex += 1
                literalOffset = 0
                fieldContentCount = 0
            }
        }
        return nil
    }

    func accept(_ tokenId: Int) {
        guard stageIndex < manifest.stages.count else {
            openTokens += 1
            return
        }
        let stage = manifest.stages[stageIndex]
        switch stage.kind {
        case "literal":
            let ids = stage.tokenIds ?? []
            if literalOffset < ids.count {
                forcedTokens += 1
                literalOffset += 1
            }
            if literalOffset >= ids.count {
                stageIndex += 1
                literalOffset = 0
            }
        case "field":
            if tokenId == manifest.newlineTokenId {
                fieldNewlineTokens += 1
                fieldTokenCounts[stage.name] = fieldContentCount
                fieldsCompleted += 1
                stageIndex += 1
                fieldContentCount = 0
            } else {
                fieldContentTokens += 1
                fieldContentCount += 1
            }
        case "open":
            openTokens += 1
        default:
            stageIndex += 1
            literalOffset = 0
            fieldContentCount = 0
        }
    }

    func stats() -> PhiStructuredCoTStats {
        let activeStage: String
        if stageIndex < manifest.stages.count {
            activeStage = manifest.stages[stageIndex].name
        } else {
            activeStage = "done"
        }
        return PhiStructuredCoTStats(
            name: manifest.name,
            forcedTokens: forcedTokens,
            fieldContentTokens: fieldContentTokens,
            fieldNewlineTokens: fieldNewlineTokens,
            openTokens: openTokens,
            fieldsCompleted: fieldsCompleted,
            activeStage: activeStage,
            fieldTokenCounts: fieldTokenCounts
        )
    }
}

struct PhiServeResponse: Encodable {
    let ok: Bool
    let generatedIds: [Int]?
    let timing: PhiServeTiming?
    let profile: PhiServeProfile?
    let structuredCoT: PhiStructuredCoTStats?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case generatedIds = "generated_ids"
        case timing
        case profile
        case structuredCoT = "structured_cot"
        case error
    }
}

final class FP16BinaryFile {
    let data: Data
    let count: Int

    init(path: String, expectedCount: Int) throws {
        data = try Data(contentsOf: URL(fileURLWithPath: path))
        count = data.count / MemoryLayout<Float16>.size
        precondition(count == expectedCount, "\(path): got \(count) fp16 values, expected \(expectedCount)")
    }

    func writeRow(_ index: Int, dim: Int, into ptr: UnsafeMutablePointer<Float16>) {
        precondition(index >= 0 && index * dim + dim <= count, "token id out of embedding bounds")
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
            memcpy(ptr, src + index * dim, dim * MemoryLayout<Float16>.size)
        }
    }

    func writeRow(_ index: Int, dim: Int, into ptr: UnsafeMutablePointer<Float16>, channelStride: Int, tokenStride: Int, tokenSlot: Int) {
        precondition(index >= 0 && index * dim + dim <= count, "token id out of embedding bounds")
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

func parseIds(_ csv: String) -> [Int] {
    csv.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
}

func parsePromptIdsFile(_ path: String) throws -> [[Int]] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    return text.split(whereSeparator: \.isNewline).compactMap { rawLine in
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        if line.isEmpty || line.hasPrefix("#") { return nil }
        let ids = parseIds(line)
        return ids.isEmpty ? nil : ids
    }
}

func printStderr(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

func formatStructuredCoTSummary(_ stats: PhiStructuredCoTStats) -> String {
    let fields = stats.fieldTokenCounts.keys.sorted().map { key in
        "\(key)=\(stats.fieldTokenCounts[key]!)"
    }.joined(separator: " ")
    let fieldSummary = fields.isEmpty ? "none" : fields
    return "StructuredCoT: name=\(stats.name) forced_tokens=\(stats.forcedTokens) field_content_tokens=\(stats.fieldContentTokens) field_newline_tokens=\(stats.fieldNewlineTokens) open_tokens=\(stats.openTokens) fields_completed=\(stats.fieldsCompleted) active_stage=\(stats.activeStage) fields=\(fieldSummary)"
}

func fillRoPE(cosPtr: UnsafeMutablePointer<Float16>, sinPtr: UnsafeMutablePointer<Float16>, pos: Int, dHalf: Int, base: Double) {
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

func copyHiddenTokens(_ src: MLMultiArray, into dst: UnsafeMutablePointer<Float16>, d: Int, batchTokens: Int, dstChannelStride: Int, dstTokenStride: Int) {
    let srcPtr = src.dataPointer.assumingMemoryBound(to: Float16.self)
    let strides = src.strides.map { Int(truncating: $0) }
    let srcChannelStride = strides[1]
    let srcTokenStride = strides[2]
    for channel in 0..<d {
        for tokenSlot in 0..<batchTokens {
            dst[channel * dstChannelStride + tokenSlot * dstTokenStride] = srcPtr[channel * srcChannelStride + tokenSlot * srcTokenStride]
        }
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
                    matched = false
                    break
                }
            }
            if matched {
                return (history[matchStart + ngramSize], ngramSize, matchStart)
            }
        }
    }
    return nil
}

func validateLayerCoverage(_ layers: [LayerSpec], nLayers: Int) {
    precondition(!layers.isEmpty, "runtime manifest has no layer shards")
    var expectedStart = 0
    for spec in layers {
        precondition(spec.start == expectedStart, "layer shard coverage gap/overlap at layer \(expectedStart); got [\(spec.start),\(spec.end))")
        precondition(spec.end > spec.start, "empty layer shard [\(spec.start),\(spec.end))")
        expectedStart = spec.end
    }
    precondition(expectedStart == nLayers, "layer shards cover 0..<\(expectedStart), expected 0..<\(nLayers)")
}

func main() throws {
    var metaPath = "local-artifacts/phi4_mini_ane/phi4mini_runtime_meta_rope96_fast_20_4_6_2.json"
    var promptIds = [199999]
    var promptIdsFile: String? = nil
    var maxNew = 1
    var warmupCalls = 0
    var warmupTokenId = 199999
    var traceTokens = false
    var profile = false
    var serve = false
    var ngramProbe = false
    var ngramForce = false
    var speculative = false
    var ngramMin = 2
    var ngramMax = 8
    var structuredCoTManifestPath: String? = nil
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--meta": metaPath = args[i + 1]; i += 2
        case "--prompt-ids": promptIds = parseIds(args[i + 1]); i += 2
        case "--prompt-ids-file": promptIdsFile = args[i + 1]; i += 2
        case "--max-new": maxNew = Int(args[i + 1])!; i += 2
        case "--warmup-calls": warmupCalls = Int(args[i + 1])!; i += 2
        case "--warmup-token-id": warmupTokenId = Int(args[i + 1])!; i += 2
        case "--serve": serve = true; i += 1
        case "--trace": traceTokens = true; i += 1
        case "--profile": profile = true; i += 1
        case "--ngram-probe": ngramProbe = true; i += 1
        case "--ngram-force": ngramForce = true; i += 1
        case "--speculative": speculative = true; i += 1
        case "--ngram-min": ngramMin = Int(args[i + 1])!; i += 2
        case "--ngram-max": ngramMax = Int(args[i + 1])!; i += 2
        case "--structured-cot": structuredCoTManifestPath = "local-artifacts/phi4_mini_ane/phi4mini_structured_cot_plan.json"; i += 1
        case "--structured-cot-manifest": structuredCoTManifestPath = args[i + 1]; i += 2
        default: i += 1
        }
    }
    precondition(!promptIds.isEmpty, "--prompt-ids must not be empty")
    precondition(ngramMin > 0, "--ngram-min must be > 0")
    precondition(ngramMax >= ngramMin, "--ngram-max must be >= --ngram-min")
    precondition(!(ngramProbe && ngramForce), "--ngram-force changes generation; do not combine it with exact --ngram-probe accounting")
    precondition(!(speculative && ngramForce), "--speculative uses exact verifier output; do not combine with approximate --ngram-force")

    let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    let meta = try JSONDecoder().decode(PhiRuntimeMeta.self, from: metaData)
    var structuredCoTManifest: StructuredCoTManifest? = nil
    if let path = structuredCoTManifestPath {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        structuredCoTManifest = try JSONDecoder().decode(StructuredCoTManifest.self, from: data)
    }
    let sortedLayers = meta.layers.sorted { $0.start < $1.start }
    validateLayerCoverage(sortedLayers, nLayers: meta.nLayers)
    // RangeDim shards: --speculative uses the unified shard set; no separate speculativeVerifier
    // section is required in the manifest (it is ignored when present).
    precondition(!meta.lmHeadShards.isEmpty, "expected at least one LM-head shard")
    precondition(meta.lmHeadShards.sorted { $0.vocabStart < $1.vocabStart }.first!.vocabStart == 0, "LM-head shards must start at vocab 0")
    precondition(serve || promptIds.count + maxNew <= meta.maxSeqLen, "prompt + max-new exceeds max_seq_len")

    func status(_ message: String) {
        if serve { printStderr(message) } else { print(message) }
    }

    status("Loading \(meta.modelFamily): \(meta.nLayers)L d=\(meta.dModel) vocab=\(meta.vocabSize) layer_shards=\(sortedLayers.count)")
    if let manifest = structuredCoTManifest {
        status("StructuredCoT: manifest=\(manifest.name) stages=\(manifest.stages.count)")
    }
    let embedPath = resolvePath(meta.embedBin, relativeTo: metaPath)
    let embed = try FP16BinaryFile(path: embedPath, expectedCount: meta.vocabSize * meta.dModel)

    guard #available(macOS 15.0, *) else {
        print("ERROR: stateful CoreML models require macOS 15+")
        return
    }

    let cfg = MLModelConfiguration()
    cfg.computeUnits = .all

    status("Loading layer shards...")
    var layerModels = [MLModel](); var layerStates = [MLState]()
    for spec in sortedLayers {
        let path = resolvePath(spec.path, relativeTo: metaPath)
        let model = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg)
        layerModels.append(model)
        layerStates.append(model.makeState())
    }
    status("Loading LM-head shards...")
    let headSpecs = meta.lmHeadShards.sorted { $0.shardIdx < $1.shardIdx }
    var headModels = [MLModel]()
    for spec in headSpecs {
        let path = resolvePath(spec.mlmodelc, relativeTo: metaPath)
        headModels.append(try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: cfg))
    }

    // RangeDim: one model set handles T=1 decode and T=2..4 chunked prefill / speculation.
    // verifierLayerModels aliases layerModels; forwardVerifier uses the same loaded shards.
    let verifierLayerModels = layerModels
    let verifierLayerStates = layerStates

    let d = meta.dModel
    let ropeHalf = (meta.ropeDim ?? meta.dHead) / 2
    let xArr = try MLMultiArray(shape: [1, NSNumber(value: d), 1, 1], dataType: .float16)
    let xPtr = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let cosArr = try MLMultiArray(shape: [1, NSNumber(value: ropeHalf)], dataType: .float16)
    let cosPtr = cosArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let sinArr = try MLMultiArray(shape: [1, NSNumber(value: ropeHalf)], dataType: .float16)
    let sinPtr = sinArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let attnMaskArr = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: meta.maxSeqLen)], dataType: .float16)
    let attnMaskPtr = attnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    for j in 0..<meta.maxSeqLen { attnMaskPtr[j] = Float16(-10000.0) }
    let kvWriteMaskArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.maxSeqLen), 1], dataType: .float16)
    let kvWriteMaskPtr = kvWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    for j in 0..<meta.maxSeqLen { kvWriteMaskPtr[j] = 0 }
    let headInputArr = try MLMultiArray(shape: [1, NSNumber(value: d), 1, 1], dataType: .float16)
    let headInputPtr = headInputArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let headStride1 = Int(truncating: headInputArr.strides[1])
    // Dragon Book strength reduction/allocation hoisting: reuse providers whose
    // MLMultiArray storage is mutated in place for each token/layer.
    let layerProvider = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: xArr),
        "rope_cos": MLFeatureValue(multiArray: cosArr),
        "rope_sin": MLFeatureValue(multiArray: sinArr),
        "attn_mask": MLFeatureValue(multiArray: attnMaskArr),
        "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr),
    ])
    let headProvider = try MLDictionaryFeatureProvider(dictionary: [
        "hidden": MLFeatureValue(multiArray: headInputArr),
    ])
    let verifierBatchTokens = meta.speculativeVerifier?.batchTokens ?? 4
    let verifierXArr = try MLMultiArray(shape: [1, NSNumber(value: d), NSNumber(value: verifierBatchTokens), 1], dataType: .float16)
    let verifierXPtr = verifierXArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let verifierXStride1 = Int(truncating: verifierXArr.strides[1])
    let verifierXStride2 = Int(truncating: verifierXArr.strides[2])
    let verifierCosArr = try MLMultiArray(shape: [NSNumber(value: verifierBatchTokens), NSNumber(value: ropeHalf)], dataType: .float16)
    let verifierCosPtr = verifierCosArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let verifierCosStride0 = Int(truncating: verifierCosArr.strides[0])
    let verifierSinArr = try MLMultiArray(shape: [NSNumber(value: verifierBatchTokens), NSNumber(value: ropeHalf)], dataType: .float16)
    let verifierSinPtr = verifierSinArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let verifierSinStride0 = Int(truncating: verifierSinArr.strides[0])
    let verifierAttnMaskArr = try MLMultiArray(shape: [1, 1, NSNumber(value: verifierBatchTokens), NSNumber(value: meta.maxSeqLen)], dataType: .float16)
    let verifierAttnMaskPtr = verifierAttnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let verifierAttnMaskStride2 = Int(truncating: verifierAttnMaskArr.strides[2])
    let verifierAttnMaskStride3 = Int(truncating: verifierAttnMaskArr.strides[3])
    let verifierKVWriteMaskArr = try MLMultiArray(shape: [1, 1, NSNumber(value: meta.maxSeqLen), NSNumber(value: verifierBatchTokens)], dataType: .float16)
    let verifierKVWriteMaskPtr = verifierKVWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
    let verifierKVWriteMaskStride2 = Int(truncating: verifierKVWriteMaskArr.strides[2])
    let verifierKVWriteMaskStride3 = Int(truncating: verifierKVWriteMaskArr.strides[3])
    let verifierLayerProvider = try MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: verifierXArr),
        "rope_cos": MLFeatureValue(multiArray: verifierCosArr),
        "rope_sin": MLFeatureValue(multiArray: verifierSinArr),
        "attn_mask": MLFeatureValue(multiArray: verifierAttnMaskArr),
        "kv_write_mask": MLFeatureValue(multiArray: verifierKVWriteMaskArr),
    ])
    _ = try MLDictionaryFeatureProvider(dictionary: [
        "hidden": MLFeatureValue(multiArray: verifierXArr),
    ])
    // Iverson/APL partitioning: independent vocab slices are whole-array
    // primitives; reduce their local maxima to one global argmax.
    let headQueue = DispatchQueue(label: "phi4mini.lm_head", attributes: .concurrent)

    func resetMasks() {
        for j in 0..<meta.maxSeqLen {
            attnMaskPtr[j] = Float16(-10000.0)
            kvWriteMaskPtr[j] = 0
        }
    }

    var tEmbed: Double = 0
    var tRopeMask: Double = 0
    var tLayers: Double = 0
    var tHeadCopy: Double = 0
    var tHeadPredictReduce: Double = 0
    var tHeadPredictShardWork: Double = 0
    var tHeadReduceShardWork: Double = 0
    var layerTimes = [Double](repeating: 0, count: layerModels.count)
    var profileCalls = 0
    var ngramTargets = 0
    var ngramProposals = 0
    var ngramAccepted = 0
    var ngramProposalBySize = [Int: Int]()
    var ngramAcceptedBySize = [Int: Int]()
    var ngramForceTargets = 0
    var ngramForced = 0
    var ngramForcedBySize = [Int: Int]()
    var specCalls = 0
    var specDrafted = 0
    var specAccepted = 0
    var specFallbacks = 0
    func resetProfileCounters() {
        tEmbed = 0
        tRopeMask = 0
        tLayers = 0
        tHeadCopy = 0
        tHeadPredictReduce = 0
        tHeadPredictShardWork = 0
        tHeadReduceShardWork = 0
        layerTimes = [Double](repeating: 0, count: layerModels.count)
        profileCalls = 0
    }

    func resetNGramCounters() {
        ngramTargets = 0
        ngramProposals = 0
        ngramAccepted = 0
        ngramProposalBySize.removeAll(keepingCapacity: true)
        ngramAcceptedBySize.removeAll(keepingCapacity: true)
        ngramForceTargets = 0
        ngramForced = 0
        ngramForcedBySize.removeAll(keepingCapacity: true)
        specCalls = 0
        specDrafted = 0
        specAccepted = 0
        specFallbacks = 0
    }

    func recordNGramProbe(history: [Int], actualNext: Int) {
        ngramTargets += 1
        guard let proposal = findNGramProposal(history: history, minN: ngramMin, maxN: ngramMax) else { return }
        ngramProposals += 1
        ngramProposalBySize[proposal.ngramSize, default: 0] += 1
        if proposal.token == actualNext {
            ngramAccepted += 1
            ngramAcceptedBySize[proposal.ngramSize, default: 0] += 1
        }
    }

    func formatNGramSummary(label: String, targets: Int, proposals: Int, accepted: Int, proposalBySize: [Int: Int], acceptedBySize: [Int: Int]) -> String {
        let proposalRate = Double(proposals) / Double(max(1, targets))
        let acceptanceRate = Double(accepted) / Double(max(1, proposals))
        let acceptedPerTarget = Double(accepted) / Double(max(1, targets))
        let sizes = Set(proposalBySize.keys).union(acceptedBySize.keys).sorted()
        let bySize = sizes.map { size in
            let acceptedForSize = acceptedBySize[size, default: 0]
            let proposedForSize = proposalBySize[size, default: 0]
            return "N\(size)=\(acceptedForSize)/\(proposedForSize)"
        }.joined(separator: " ")
        let bySizeSummary = bySize.isEmpty ? "none" : bySize
        return "\(label): targets=\(targets) proposals=\(proposals) accepted=\(accepted) proposal_rate=\(String(format: "%.3f", proposalRate)) acceptance_rate=\(String(format: "%.3f", acceptanceRate)) accepted_per_target=\(String(format: "%.3f", acceptedPerTarget)) min_n=\(ngramMin) max_n=\(ngramMax) by_n=\(bySizeSummary)"
    }

    func ngramProbeSummary() -> String {
        formatNGramSummary(label: "NGramProbe", targets: ngramTargets, proposals: ngramProposals, accepted: ngramAccepted, proposalBySize: ngramProposalBySize, acceptedBySize: ngramAcceptedBySize)
    }

    func ngramForceConstraint(history: [Int]) -> TokenConstraint? {
        guard ngramForce else { return nil }
        ngramForceTargets += 1
        guard let proposal = findNGramProposal(history: history, minN: ngramMin, maxN: ngramMax) else { return nil }
        ngramForced += 1
        ngramForcedBySize[proposal.ngramSize, default: 0] += 1
        return TokenConstraint(forcedToken: proposal.token, reason: "ngram.N\(proposal.ngramSize)")
    }

    func ngramForceSummary() -> String {
        let forceRate = Double(ngramForced) / Double(max(1, ngramForceTargets))
        let bySize = ngramForcedBySize.keys.sorted().map { size in
            "N\(size)=\(ngramForcedBySize[size, default: 0])"
        }.joined(separator: " ")
        let bySizeSummary = bySize.isEmpty ? "none" : bySize
        return "NGramForce: targets=\(ngramForceTargets) forced=\(ngramForced) force_rate=\(String(format: "%.3f", forceRate)) min_n=\(ngramMin) max_n=\(ngramMax) by_n=\(bySizeSummary)"
    }

    func speculativeSummary() -> String {
        let acceptRate = Double(specAccepted) / Double(max(1, specDrafted))
        let avgDrafted = Double(specDrafted) / Double(max(1, specCalls))
        return "Speculative: calls=\(specCalls) drafted=\(specDrafted) accepted=\(specAccepted) fallbacks=\(specFallbacks) accept_rate=\(String(format: "%.3f", acceptRate)) avg_drafted_per_call=\(String(format: "%.3f", avgDrafted)) batch_tokens=\(verifierBatchTokens)"
    }

    func formatNGramForceSummary(label: String, targets: Int, forced: Int, forcedBySize: [Int: Int]) -> String {
        let forceRate = Double(forced) / Double(max(1, targets))
        let bySize = forcedBySize.keys.sorted().map { size in
            "N\(size)=\(forcedBySize[size, default: 0])"
        }.joined(separator: " ")
        let bySizeSummary = bySize.isEmpty ? "none" : bySize
        return "\(label): targets=\(targets) forced=\(forced) force_rate=\(String(format: "%.3f", forceRate)) min_n=\(ngramMin) max_n=\(ngramMax) by_n=\(bySizeSummary)"
    }

    func profileSnapshot() -> PhiServeProfile {
        PhiServeProfile(
            calls: profileCalls,
            embedSeconds: tEmbed,
            ropeMaskSeconds: tRopeMask,
            layersSeconds: tLayers,
            headCopySeconds: tHeadCopy,
            headPredictReduceSeconds: tHeadPredictReduce,
            headPredictShardWorkSeconds: tHeadPredictShardWork,
            headReduceShardWorkSeconds: tHeadReduceShardWork
        )
    }

    func fillVerifierRoPE(row: Int, pos: Int) {
        for j in 0..<ropeHalf {
            let inv = 1.0 / pow(meta.ropeFreqBase, Double(j) / Double(ropeHalf))
            let angle = Double(pos) * inv
            verifierCosPtr[row * verifierCosStride0 + j] = Float16(cos(angle))
            verifierSinPtr[row * verifierSinStride0 + j] = Float16(sin(angle))
        }
    }

    func resetVerifierInputs() {
        for channel in 0..<d {
            for tokenSlot in 0..<verifierBatchTokens {
                verifierXPtr[channel * verifierXStride1 + tokenSlot * verifierXStride2] = 0
            }
        }
        for tokenSlot in 0..<verifierBatchTokens {
            for j in 0..<meta.maxSeqLen {
                verifierAttnMaskPtr[tokenSlot * verifierAttnMaskStride2 + j * verifierAttnMaskStride3] = Float16(-10000.0)
                verifierKVWriteMaskPtr[j * verifierKVWriteMaskStride2 + tokenSlot * verifierKVWriteMaskStride3] = 0
            }
        }
    }

    func predictBatchHead(models: [MLModel], specs: [LMHeadShardSpec], provider: MLFeatureProvider, batchTokens: Int) throws -> [Int] {
        let group = DispatchGroup()
        let lock = NSLock()
        let slots = models.count * batchTokens
        var shardBestLogit = [Float](repeating: -Float.infinity, count: slots)
        var shardBestToken = [Int](repeating: -1, count: slots)
        var shardError: Error? = nil
        for s in 0..<models.count {
            group.enter()
            headQueue.async {
                do {
                    let result = try models[s].prediction(from: provider)
                    let logits = result.featureValue(for: "logits")!.multiArrayValue!
                    let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                    let strides = logits.strides.map { Int(truncating: $0) }
                    let vocabStride = strides[1]
                    let tokenStride = strides[2]
                    let spec = specs[s]
                    var localLogits = [Float](repeating: -Float.infinity, count: batchTokens)
                    var localTokens = [Int](repeating: -1, count: batchTokens)
                    for local in 0..<(spec.vocabEnd - spec.vocabStart) {
                        let token = spec.vocabStart + local
                        let base = local * vocabStride
                        for tokenSlot in 0..<batchTokens {
                            let value = Float(ptr[base + tokenSlot * tokenStride])
                            if value > localLogits[tokenSlot] {
                                localLogits[tokenSlot] = value
                                localTokens[tokenSlot] = token
                            }
                        }
                    }
                    lock.lock()
                    for tokenSlot in 0..<batchTokens {
                        let idx = s * batchTokens + tokenSlot
                        shardBestLogit[idx] = localLogits[tokenSlot]
                        shardBestToken[idx] = localTokens[tokenSlot]
                    }
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
        var best = [Int](repeating: -1, count: batchTokens)
        for tokenSlot in 0..<batchTokens {
            var bestLogit = -Float.infinity
            var bestToken = -1
            for s in 0..<models.count {
                let idx = s * batchTokens + tokenSlot
                if shardBestToken[idx] >= 0 && shardBestLogit[idx] > bestLogit {
                    bestLogit = shardBestLogit[idx]
                    bestToken = shardBestToken[idx]
                }
            }
            if bestToken < 0 {
                throw NSError(domain: "Phi4MiniANE", code: 3, userInfo: [NSLocalizedDescriptionKey: "batch LM head produced no token for slot \(tokenSlot)"])
            }
            best[tokenSlot] = bestToken
        }
        return best
    }

    // Predict the next token for each of `count` batch slots from the current verifierXArr output,
    // using the T=1 LM-head shards sequentially per slot (one concurrent head call per slot).
    // tokenConstraint is applied only to the final slot (the only prediction that matters for
    // chunked prefill; for speculative decode all slots are used but only the last may be constrained).
    func predictSlotsWithT1Head(count: Int, tokenConstraint: TokenConstraint? = nil) throws -> [Int] {
        var results = [Int](repeating: -1, count: count)
        for slot in 0..<count {
            for channel in 0..<d {
                headInputPtr[channel * headStride1] =
                    verifierXPtr[channel * verifierXStride1 + slot * verifierXStride2]
            }
            let slotConstraint: TokenConstraint? = (slot == count - 1) ? tokenConstraint : nil
            let group = DispatchGroup()
            let lock = NSLock()
            var shardBestLogit = [Float](repeating: -Float.infinity, count: headModels.count)
            var shardBestToken = [Int](repeating: -1, count: headModels.count)
            var shardError: Error? = nil
            for s in 0..<headModels.count {
                group.enter()
                headQueue.async {
                    do {
                        let result = try headModels[s].prediction(from: headProvider)
                        let logits = result.featureValue(for: "logits")!.multiArrayValue!
                        let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                        let stride1 = Int(truncating: logits.strides[1])
                        let spec = headSpecs[s]
                        var localBestLogit = -Float.infinity
                        var localBestToken = -1
                        for local in 0..<(spec.vocabEnd - spec.vocabStart) {
                            let token = spec.vocabStart + local
                            if let slotConstraint, !slotConstraint.allows(token) { continue }
                            let value = Float(ptr[local * stride1])
                            if value > localBestLogit { localBestLogit = value; localBestToken = token }
                        }
                        lock.lock()
                        shardBestLogit[s] = localBestLogit
                        shardBestToken[s] = localBestToken
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
            var bestLogit = -Float.infinity
            var bestToken = -1
            for s in 0..<headModels.count {
                if shardBestToken[s] >= 0 && shardBestLogit[s] > bestLogit {
                    bestLogit = shardBestLogit[s]; bestToken = shardBestToken[s]
                }
            }
            if bestToken < 0 {
                throw NSError(domain: "Phi4MiniANE", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "T=1 LM-head produced no token for batch slot \(slot)"])
            }
            results[slot] = bestToken
        }
        return results
    }

    // advanceOnly=true: run embed + RoPE + verifier layers to populate KV cache
    // but skip the LM-head.  Used for non-final prefill chunks where only the KV
    // update matters.  Returns empty array when advanceOnly.
    func forwardVerifier(tokens: [Int], posStart: Int, cacheSeqLen: Int, states: [MLState], collectProfile: Bool = true, advanceOnly: Bool = false, tokenConstraint: TokenConstraint? = nil) throws -> [Int] {
        precondition(tokens.count >= 1 && tokens.count <= verifierBatchTokens, "invalid verifier token count")
        let t0 = CFAbsoluteTimeGetCurrent()
        resetVerifierInputs()
        for tokenSlot in 0..<verifierBatchTokens {
            fillVerifierRoPE(row: tokenSlot, pos: posStart + tokenSlot)
        }
        for tokenSlot in 0..<tokens.count {
            embed.writeRow(tokens[tokenSlot], dim: d, into: verifierXPtr, channelStride: verifierXStride1, tokenStride: verifierXStride2, tokenSlot: tokenSlot)
            let visibleEnd = min(meta.maxSeqLen - 1, cacheSeqLen + tokenSlot)
            if visibleEnd >= 0 {
                for j in 0...visibleEnd {
                    verifierAttnMaskPtr[tokenSlot * verifierAttnMaskStride2 + j * verifierAttnMaskStride3] = 0
                }
            }
            verifierKVWriteMaskPtr[(cacheSeqLen + tokenSlot) * verifierKVWriteMaskStride2 + tokenSlot * verifierKVWriteMaskStride3] = Float16(1.0)
        }
        let t1 = CFAbsoluteTimeGetCurrent()
        for idx in 0..<verifierLayerModels.count {
            let lt0 = CFAbsoluteTimeGetCurrent()
            try autoreleasepool {
                let result = try verifierLayerModels[idx].prediction(from: verifierLayerProvider, using: states[idx])
                let output = result.featureValue(for: "hidden")!.multiArrayValue!
                copyHiddenTokens(output, into: verifierXPtr, d: d, batchTokens: verifierBatchTokens, dstChannelStride: verifierXStride1, dstTokenStride: verifierXStride2)
            }
            let lt1 = CFAbsoluteTimeGetCurrent()
            if collectProfile && idx < layerTimes.count { layerTimes[idx] += lt1 - lt0 }
        }
        let t2 = CFAbsoluteTimeGetCurrent()
        // Chunked-prefill advance-only: KV cache written; skip LM head entirely.
        if advanceOnly {
            if collectProfile { tEmbed += t1 - t0; tLayers += t2 - t1; profileCalls += 1 }
            return []
        }
        let predictions = try predictSlotsWithT1Head(count: tokens.count, tokenConstraint: tokenConstraint)
        let t3 = CFAbsoluteTimeGetCurrent()
        if collectProfile {
            tEmbed += t1 - t0
            tLayers += t2 - t1
            tHeadPredictReduce += t3 - t2
            profileCalls += 1
        }
        if traceTokens && collectProfile {
            print("spec pos=\(posStart) in=\(tokens) preds=\(Array(predictions.prefix(tokens.count)))")
        }
        return predictions
    }

    // advanceOnly=true: run embed + RoPE + all layer shards to populate KV cache,
    // but skip the LM-head projection entirely.  Used for non-final prefill tokens
    // where only the KV cache update matters, not the predicted token.
    // Returns -1 when advanceOnly; caller must ignore the return value.
    func forwardOne(tokenId: Int, pos: Int, cacheSeqLen: inout Int, states: [MLState], collectProfile: Bool = true, tokenConstraint: TokenConstraint? = nil, advanceOnly: Bool = false) throws -> Int {
        let t0 = CFAbsoluteTimeGetCurrent()
        embed.writeRow(tokenId, dim: d, into: xPtr)
        let t1 = CFAbsoluteTimeGetCurrent()
        fillRoPE(cosPtr: cosPtr, sinPtr: sinPtr, pos: pos, dHalf: ropeHalf, base: meta.ropeFreqBase)
        attnMaskPtr[cacheSeqLen] = 0
        if cacheSeqLen > 0 { kvWriteMaskPtr[cacheSeqLen - 1] = 0 }
        kvWriteMaskPtr[cacheSeqLen] = Float16(1.0)
        let t2 = CFAbsoluteTimeGetCurrent()

        for idx in 0..<layerModels.count {
            let lt0 = CFAbsoluteTimeGetCurrent()
            // Wrap prediction + copy in autoreleasepool so the MLFeatureProvider
            // (and its ANE output-port binding) is released immediately after we
            // copy the data. Without this, ARC defers the release and the ANE
            // output-port pool (~10 slots) is exhausted after ~9 requests.
            try autoreleasepool {
                let result = try layerModels[idx].prediction(from: layerProvider, using: states[idx])
                let output = result.featureValue(for: "hidden")!.multiArrayValue!
                copyFlatFloat16(output, into: xPtr, count: d)
            }
            let lt1 = CFAbsoluteTimeGetCurrent()
            if collectProfile { layerTimes[idx] += lt1 - lt0 }
        }
        let t3 = CFAbsoluteTimeGetCurrent()
        cacheSeqLen += 1

        // Prefill advance-only: KV cache written; skip LM head entirely.
        if advanceOnly {
            if collectProfile {
                tEmbed += t1 - t0
                tRopeMask += t2 - t1
                tLayers += t3 - t2
                profileCalls += 1
            }
            return -1
        }

        if let forcedToken = tokenConstraint?.forcedToken {
            if collectProfile {
                tEmbed += t1 - t0
                tRopeMask += t2 - t1
                tLayers += t3 - t2
                profileCalls += 1
            }
            if traceTokens && collectProfile {
                let reason = tokenConstraint?.reason ?? "constraint"
                print("pos=\(pos) in=\(tokenId) next=\(forcedToken) forced=\(reason)")
            }
            return forcedToken
        }

        for j in 0..<d { headInputPtr[j * headStride1] = xPtr[j] }
        let t4 = CFAbsoluteTimeGetCurrent()

        let group = DispatchGroup()
        let lock = NSLock()
        var shardBestLogit = [Float](repeating: -Float.infinity, count: headModels.count)
        var shardBestToken = [Int](repeating: 0, count: headModels.count)
        var shardPredictWork = [Double](repeating: 0, count: headModels.count)
        var shardReduceWork = [Double](repeating: 0, count: headModels.count)
        var shardError: Error? = nil
        for s in 0..<headModels.count {
            group.enter()
            headQueue.async {
                let hp0 = CFAbsoluteTimeGetCurrent()
                var localBestLogit = -Float.infinity
                var localBestToken = -1
                var predictWork = 0.0
                var reduceWork = 0.0
                var capturedError: Error? = nil
                autoreleasepool {
                    do {
                        let result = try headModels[s].prediction(from: headProvider)
                        let hp1 = CFAbsoluteTimeGetCurrent()
                        let logits = result.featureValue(for: "logits")!.multiArrayValue!
                        let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
                        let stride1 = Int(truncating: logits.strides[1])
                        let spec = headSpecs[s]
                        for local in 0..<(spec.vocabEnd - spec.vocabStart) {
                            let token = spec.vocabStart + local
                            if let tokenConstraint, !tokenConstraint.allows(token) { continue }
                            let value = Float(ptr[local * stride1])
                            if value > localBestLogit {
                                localBestLogit = value
                                localBestToken = token
                            }
                        }
                        let hr1 = CFAbsoluteTimeGetCurrent()
                        predictWork = hp1 - hp0
                        reduceWork = hr1 - hp1
                    } catch {
                        capturedError = error
                    }
                }
                lock.lock()
                if let err = capturedError {
                    if shardError == nil { shardError = err }
                } else {
                    shardBestLogit[s] = localBestLogit
                    shardBestToken[s] = localBestToken
                    shardPredictWork[s] = predictWork
                    shardReduceWork[s] = reduceWork
                }
                lock.unlock()
                group.leave()
            }
        }
        group.wait()
        if let error = shardError { throw error }
        var bestLogit = -Float.infinity
        var bestToken = -1
        for s in 0..<headModels.count {
            if shardBestToken[s] >= 0 && shardBestLogit[s] > bestLogit {
                bestLogit = shardBestLogit[s]
                bestToken = shardBestToken[s]
            }
        }
        if bestToken < 0 {
            throw NSError(domain: "Phi4MiniANE", code: 2, userInfo: [NSLocalizedDescriptionKey: "token constraint produced no valid candidates"])
        }
        let t5 = CFAbsoluteTimeGetCurrent()
        if collectProfile {
            tEmbed += t1 - t0
            tRopeMask += t2 - t1
            tLayers += t3 - t2
            tHeadCopy += t4 - t3
            tHeadPredictReduce += t5 - t4
            tHeadPredictShardWork += shardPredictWork.reduce(0, +)
            tHeadReduceShardWork += shardReduceWork.reduce(0, +)
            profileCalls += 1
        }
        if traceTokens && collectProfile {
            print("pos=\(pos) in=\(tokenId) next=\(bestToken) logit=\(bestLogit)")
        }
        return bestToken
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

    func runGenerationSpeculative(promptIds requestPromptIds: [Int], maxNew requestMaxNew: Int, states: [MLState], requestProfile: Bool, requestStructuredCoT: Bool) throws -> (generated: [Int], timing: PhiServeTiming, profile: PhiServeProfile?, structuredCoT: PhiStructuredCoTStats?) {
        precondition(!requestPromptIds.isEmpty, "prompt_ids must not be empty")
        precondition(requestPromptIds.count + requestMaxNew <= meta.maxSeqLen, "prompt + max_new exceeds max_seq_len")
        precondition(!requestStructuredCoT, "--speculative does not yet combine with structured-CoT constraints")
        resetProfileCounters()
        resetNGramCounters()
        var generated = [Int]()
        var cacheSeqLen = 0
        var forwardCalls = 0
        var next = requestPromptIds[0]

        let prefillStart = CFAbsoluteTimeGetCurrent()
        // Iverson APL batched inner-product: process verifierBatchTokens (T=4) prefill
        // tokens per call instead of 1, reducing layer passes from N to ceil(N/T).
        // Non-final chunks skip the LM head (advanceOnly); only the last chunk
        // runs the head to produce the first generated token.
        var i = 0
        while i < requestPromptIds.count {
            let chunkEnd = min(i + verifierBatchTokens, requestPromptIds.count)
            let chunk = Array(requestPromptIds[i..<chunkEnd])
            let isLastChunk = (chunkEnd == requestPromptIds.count)
            if isLastChunk {
                let predictions = try forwardVerifier(tokens: chunk, posStart: i, cacheSeqLen: cacheSeqLen, states: states)
                next = predictions[chunk.count - 1]
            } else {
                _ = try forwardVerifier(tokens: chunk, posStart: i, cacheSeqLen: cacheSeqLen, states: states, advanceOnly: true)
            }
            cacheSeqLen += chunk.count
            forwardCalls += 1
            i += chunk.count
        }
        let prefillElapsed = CFAbsoluteTimeGetCurrent() - prefillStart
        generated.append(next)
        if requestProfile { resetProfileCounters() }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        while generated.count < requestMaxNew {
            let history = requestPromptIds + generated
            let inputToken = generated.last!
            let draft = speculativeDraft(history: history, firstToken: inputToken)
            let predictions = try forwardVerifier(tokens: draft, posStart: requestPromptIds.count + generated.count - 1, cacheSeqLen: cacheSeqLen, states: states)
            forwardCalls += 1
            specCalls += 1
            specDrafted += max(0, draft.count - 1)

            var acceptedContinuation = 0
            var emittedFallback = false
            if draft.count > 1 {
                for idx in 1..<draft.count {
                    if predictions[idx - 1] == draft[idx] {
                        generated.append(draft[idx])
                        acceptedContinuation += 1
                        specAccepted += 1
                        if generated.count >= requestMaxNew { break }
                    } else {
                        generated.append(predictions[idx - 1])
                        emittedFallback = true
                        specFallbacks += 1
                        break
                    }
                }
            }
            if !emittedFallback && generated.count < requestMaxNew {
                generated.append(predictions[draft.count - 1])
            }

            cacheSeqLen += 1 + acceptedContinuation
            if generated.last == meta.eosTokenId || generated.last == meta.bosTokenId { break }
        }
        let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
        let totalForwardElapsed = prefillElapsed + decodeElapsed
        let decodeTokens = max(0, generated.count - 1)
        let decodeTokPerSec = decodeElapsed > 0 && decodeTokens > 0 ? Double(decodeTokens) / decodeElapsed : 0
        let forwardTokPerSec = totalForwardElapsed > 0 ? Double(forwardCalls) / totalForwardElapsed : 0
        let timing = PhiServeTiming(
            prefillTokens: requestPromptIds.count,
            prefillSeconds: prefillElapsed,
            decodeTokens: decodeTokens,
            decodeSeconds: decodeElapsed,
            decodeTokensPerSecond: decodeTokPerSec,
            forwardCalls: forwardCalls,
            forwardSeconds: totalForwardElapsed,
            forwardTokensPerSecond: forwardTokPerSec
        )
        return (generated, timing, requestProfile ? profileSnapshot() : nil, nil)
    }

    func runGeneration(promptIds requestPromptIds: [Int], maxNew requestMaxNew: Int, states: [MLState], requestProfile: Bool, requestStructuredCoT: Bool) throws -> (generated: [Int], timing: PhiServeTiming, profile: PhiServeProfile?, structuredCoT: PhiStructuredCoTStats?) {
        if speculative {
            return try runGenerationSpeculative(promptIds: requestPromptIds, maxNew: requestMaxNew, states: verifierLayerStates, requestProfile: requestProfile, requestStructuredCoT: requestStructuredCoT)
        }
        precondition(!requestPromptIds.isEmpty, "prompt_ids must not be empty")
        precondition(requestPromptIds.count + requestMaxNew <= meta.maxSeqLen, "prompt + max_new exceeds max_seq_len")
        precondition(!requestStructuredCoT || structuredCoTManifest != nil, "--structured-cot requires --structured-cot-manifest or the default manifest")
        resetMasks()
        resetProfileCounters()
        resetNGramCounters()
        let structuredSampler = requestStructuredCoT ? StructuredCoTSampler(manifest: structuredCoTManifest!) : nil
        var generated = [Int]()
        var next = requestPromptIds[0]
        var forwardCalls = 0
        var cacheSeqLen = 0

        // Chunked prefill: process up to verifierBatchTokens tokens per ANE call,
        // reducing layer passes from N to ceil(N / verifierBatchTokens).
        // Iverson APL inner-product principle (BOOK_ANALYSIS.md Exp 26).
        // Single-token chunks always use forwardOne (T=1 arrays) to avoid triggering
        // a T=4 JIT recompilation on first call — critical for short prompts / decode.
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var prefillIdx = 0
        var usedBatchPrefill = false
        while prefillIdx < requestPromptIds.count {
            let chunkEnd = min(prefillIdx + verifierBatchTokens, requestPromptIds.count)
            let chunk = Array(requestPromptIds[prefillIdx..<chunkEnd])
            let isLastChunk = (chunkEnd == requestPromptIds.count)
            if chunk.count == 1 {
                // T=1 path: reuse the pre-allocated T=1 arrays; no extra JIT
                let token = chunk[0]
                if isLastChunk {
                    let structuredConstraint = structuredSampler?.constraintForNextToken()
                    let forceConstraint = structuredConstraint == nil ? ngramForceConstraint(history: requestPromptIds) : nil
                    let constraint = structuredConstraint ?? forceConstraint
                    next = try forwardOne(tokenId: token, pos: prefillIdx, cacheSeqLen: &cacheSeqLen, states: states, tokenConstraint: constraint)
                } else {
                    _ = try forwardOne(tokenId: token, pos: prefillIdx, cacheSeqLen: &cacheSeqLen, states: states, advanceOnly: true)
                }
            } else {
                // T=2..4 batch path: uses verifier arrays for chunked prefill throughput
                usedBatchPrefill = true
                if isLastChunk {
                    let structuredConstraint = structuredSampler?.constraintForNextToken()
                    let forceConstraint = structuredConstraint == nil ? ngramForceConstraint(history: requestPromptIds) : nil
                    let constraint = structuredConstraint ?? forceConstraint
                    let predictions = try forwardVerifier(tokens: chunk, posStart: prefillIdx,
                        cacheSeqLen: cacheSeqLen, states: states, tokenConstraint: constraint)
                    next = predictions[chunk.count - 1]
                } else {
                    _ = try forwardVerifier(tokens: chunk, posStart: prefillIdx,
                        cacheSeqLen: cacheSeqLen, states: states, advanceOnly: true)
                }
                cacheSeqLen += chunk.count
            }
            forwardCalls += 1
            prefillIdx += chunk.count
        }
        // If any batch prefill was done, prime the T=1 attn mask for decode:
        // forwardVerifier uses verifierAttnMaskPtr (not attnMaskPtr).
        if usedBatchPrefill {
            for j in 0..<cacheSeqLen { attnMaskPtr[j] = 0 }
        }
        let prefillElapsed = CFAbsoluteTimeGetCurrent() - prefillStart
        structuredSampler?.accept(next)
        if ngramProbe { recordNGramProbe(history: requestPromptIds, actualNext: next) }
        generated.append(next)
        if requestProfile { resetProfileCounters() }

        let decodeStart = CFAbsoluteTimeGetCurrent()
        for step in 1..<requestMaxNew {
            let probeHistory = ngramProbe ? requestPromptIds + generated : []
            let generationHistory = requestPromptIds + generated
            let structuredConstraint = structuredSampler?.constraintForNextToken()
            let forceConstraint = structuredConstraint == nil ? ngramForceConstraint(history: generationHistory) : nil
            let constraint = structuredConstraint ?? forceConstraint
            next = try forwardOne(tokenId: generated.last!, pos: requestPromptIds.count + step - 1, cacheSeqLen: &cacheSeqLen, states: states, tokenConstraint: constraint)
            forwardCalls += 1
            structuredSampler?.accept(next)
            if ngramProbe { recordNGramProbe(history: probeHistory, actualNext: next) }
            generated.append(next)
            if next == meta.eosTokenId || next == meta.bosTokenId { break }
        }
        let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
        let totalForwardElapsed = prefillElapsed + decodeElapsed
        let decodeTokens = max(0, generated.count - 1)
        let decodeTokPerSec = decodeElapsed > 0 && decodeTokens > 0 ? Double(decodeTokens) / decodeElapsed : 0
        let forwardTokPerSec = totalForwardElapsed > 0 ? Double(forwardCalls) / totalForwardElapsed : 0
        let timing = PhiServeTiming(
            prefillTokens: requestPromptIds.count,
            prefillSeconds: prefillElapsed,
            decodeTokens: decodeTokens,
            decodeSeconds: decodeElapsed,
            decodeTokensPerSecond: decodeTokPerSec,
            forwardCalls: forwardCalls,
            forwardSeconds: totalForwardElapsed,
            forwardTokensPerSecond: forwardTokPerSec
        )
        return (generated, timing, requestProfile ? profileSnapshot() : nil, structuredSampler?.stats())
    }

    if warmupCalls > 0 {
        status("Warmup: calls=\(warmupCalls) token_id=\(warmupTokenId) using isolated state")
        let warmupStart = CFAbsoluteTimeGetCurrent()
        var warmupStates = layerModels.map { $0.makeState() }
        var warmupCacheSeqLen = 0
        var warmToken = warmupTokenId
        resetMasks()
        for pos in 0..<warmupCalls {
            warmToken = try forwardOne(tokenId: warmToken, pos: pos, cacheSeqLen: &warmupCacheSeqLen, states: warmupStates, collectProfile: false)
        }
        warmupStates.removeAll(keepingCapacity: false)
        resetMasks()
        status("Warmup: elapsed_s=\(String(format: "%.6f", CFAbsoluteTimeGetCurrent() - warmupStart))")
    }

    if serve {
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // Allocate states once and reuse them across all serve requests.
        // resetMasks() is called at the top of runGeneration() so stale KV data
        // from the previous request is masked out — this is safe.
        let serveStates = layerModels.map { $0.makeState() }
        print("READY {\"protocol\":\"phi4mini-jsonl-v1\",\"fields\":{\"prompt_ids\":\"int[]\",\"max_new\":\"int\",\"profile\":\"bool\",\"structured_cot\":\"bool\"}}")
        fflush(stdout)
        while let line = readLine() {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            do {
                let request = try decoder.decode(PhiServeRequest.self, from: Data(line.utf8))
                let requestMaxNew = request.maxNew ?? maxNew
                let requestStructuredCoT = request.structuredCoT ?? (structuredCoTManifest != nil)
                let result = try runGeneration(promptIds: request.promptIds, maxNew: requestMaxNew, states: serveStates, requestProfile: request.profile ?? profile, requestStructuredCoT: requestStructuredCoT)
                let response = PhiServeResponse(ok: true, generatedIds: result.generated, timing: result.timing, profile: result.profile, structuredCoT: result.structuredCoT, error: nil)
                print(String(data: try encoder.encode(response), encoding: .utf8)!)
                fflush(stdout)
            } catch {
                let response = PhiServeResponse(ok: false, generatedIds: nil, timing: nil, profile: nil, structuredCoT: nil, error: String(describing: error))
                print(String(data: try encoder.encode(response), encoding: .utf8)!)
                fflush(stdout)
            }
        }
        return
    }

    let promptRuns = try promptIdsFile.map { try parsePromptIdsFile($0) } ?? [promptIds]
    precondition(!promptRuns.isEmpty, "prompt-id file produced no prompts")
    var suiteNGramTargets = 0
    var suiteNGramProposals = 0
    var suiteNGramAccepted = 0
    var suiteNGramProposalBySize = [Int: Int]()
    var suiteNGramAcceptedBySize = [Int: Int]()
    var suiteNGramForceTargets = 0
    var suiteNGramForced = 0
    var suiteNGramForcedBySize = [Int: Int]()

    for (runIndex, runPromptIds) in promptRuns.enumerated() {
        let runStates = promptRuns.count == 1 ? layerStates : layerModels.map { $0.makeState() }
        print("PromptRun: index=\(runIndex) prompt_tokens=\(runPromptIds.count)")
        print("Prompt IDs: \(runPromptIds)")
        let result = try runGeneration(promptIds: runPromptIds, maxNew: maxNew, states: runStates, requestProfile: profile, requestStructuredCoT: structuredCoTManifest != nil)
        let prefillTokPerSec = result.timing.prefillSeconds > 0 ? Double(result.timing.prefillTokens) / result.timing.prefillSeconds : 0
        print("Generated IDs: \(result.generated)")
        print("Timing: prefill_tokens=\(result.timing.prefillTokens) prefill_s=\(String(format: "%.6f", result.timing.prefillSeconds)) prefill_tok_s=\(String(format: "%.3f", prefillTokPerSec))")
        print("Timing: decode_tokens=\(result.timing.decodeTokens) decode_s=\(String(format: "%.6f", result.timing.decodeSeconds)) decode_tok_s=\(String(format: "%.3f", result.timing.decodeTokensPerSecond))")
        print("Timing: forward_calls=\(result.timing.forwardCalls) forward_s=\(String(format: "%.6f", result.timing.forwardSeconds)) forward_tok_s=\(String(format: "%.3f", result.timing.forwardTokensPerSecond))")
        if profile {
            let denom = Double(max(1, profileCalls))
            print("ProfileDecode: calls=\(profileCalls) embed_s=\(String(format: "%.6f", tEmbed)) rope_mask_s=\(String(format: "%.6f", tRopeMask)) layers_s=\(String(format: "%.6f", tLayers)) head_copy_s=\(String(format: "%.6f", tHeadCopy)) head_predict_reduce_s=\(String(format: "%.6f", tHeadPredictReduce)) head_predict_shard_work_s=\(String(format: "%.6f", tHeadPredictShardWork)) head_reduce_shard_work_s=\(String(format: "%.6f", tHeadReduceShardWork))")
            print("ProfileDecodePerToken: embed_ms=\(String(format: "%.3f", tEmbed / denom * 1000)) rope_mask_ms=\(String(format: "%.3f", tRopeMask / denom * 1000)) layers_ms=\(String(format: "%.3f", tLayers / denom * 1000)) head_copy_ms=\(String(format: "%.3f", tHeadCopy / denom * 1000)) head_predict_reduce_ms=\(String(format: "%.3f", tHeadPredictReduce / denom * 1000)) head_predict_shard_work_ms=\(String(format: "%.3f", tHeadPredictShardWork / denom * 1000)) head_reduce_shard_work_ms=\(String(format: "%.3f", tHeadReduceShardWork / denom * 1000))")
            let sortedLayerTimes = layerTimes.enumerated().sorted { $0.element > $1.element }
            let top = sortedLayerTimes.prefix(5).map { item in
                let spec = sortedLayers[item.offset]
                return "L\(spec.start)-\(spec.end)=\(String(format: "%.3f", item.element / denom * 1000))ms"
            }.joined(separator: " ")
            let layerMean = tLayers / Double(max(1, profileCalls * layerModels.count)) * 1000
            print("ProfileDecodeLayers: layer_shards=\(layerModels.count) mean_layer_shard_call_ms=\(String(format: "%.3f", layerMean)) top5=\(top)")
        }
        if let structuredCoT = result.structuredCoT {
            print(formatStructuredCoTSummary(structuredCoT))
        }
        if speculative {
            print(speculativeSummary())
        }
        if ngramProbe {
            print(ngramProbeSummary())
            suiteNGramTargets += ngramTargets
            suiteNGramProposals += ngramProposals
            suiteNGramAccepted += ngramAccepted
            for (size, count) in ngramProposalBySize { suiteNGramProposalBySize[size, default: 0] += count }
            for (size, count) in ngramAcceptedBySize { suiteNGramAcceptedBySize[size, default: 0] += count }
        }
        if ngramForce {
            print(ngramForceSummary())
            suiteNGramForceTargets += ngramForceTargets
            suiteNGramForced += ngramForced
            for (size, count) in ngramForcedBySize { suiteNGramForcedBySize[size, default: 0] += count }
        }
    }
    if ngramProbe && promptRuns.count > 1 {
        print(formatNGramSummary(label: "NGramProbeSuite", targets: suiteNGramTargets, proposals: suiteNGramProposals, accepted: suiteNGramAccepted, proposalBySize: suiteNGramProposalBySize, acceptedBySize: suiteNGramAcceptedBySize))
    }
    if ngramForce && promptRuns.count > 1 {
        print(formatNGramForceSummary(label: "NGramForceSuite", targets: suiteNGramForceTargets, forced: suiteNGramForced, forcedBySize: suiteNGramForcedBySize))
    }
}

try main()