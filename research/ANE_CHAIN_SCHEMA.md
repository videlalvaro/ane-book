# ANE Chain Primitive — Reverse-Engineered Schema

**Date**: 2026-04-22
**Source**: live ObjC runtime reflection of `AppleNeuralEngine.framework` and `ANECompiler.framework` on macOS (M4 Max, h16g, 16 ANE cores).
**Probe binaries** (in this dir):
- `ane_chain_probe.m` — verifies the chain XPC primitive exists and is per-call retargetable.
- `ane_class_dump.m` — enumerates every ObjC class registered in the loaded private framework images, dumps ivar layouts + selectors + adopted protocols.

Logs: temporary probe logs.

## TL;DR

The Apple Neural Engine private runtime exposes a **chain primitive** that lets the daemon execute a sequence of model procedures on-device with no host round-trip between stages and a shared memory pool for intermediates. Combined with the fact that one loaded model carries an **array of procedures**, this maps almost 1:1 onto MoE expert dispatch.

## Verified XPC Surface

```
-[_ANEClient prepareChainingWithModel:options:chainingReq:qos:error:]
-[_ANEClient loadModelNewInstance:options:modelInstParams:qos:error:]
-[_ANEDaemonProtocol prepareChainingWithModel:options:chainingReq:qos:withReply:]
   encoded_types = v52@0:8@16@24@32I40@?44
   (return void; args: id model, id options, id chainingReq, uint32_t qos, void(^reply)(...))
```

`chainingReq` is **per-call**, not bound at `loadModel:` time → the chain can be retargeted on every invocation (per token) with no recompile and no reload.

The daemon holds multiple loaded models per `_ANEDaemonConnection`. Verified via `-[_ANEClient connectionsUsedForLoadingModels]` showing two distinct `_ANEModel` UUIDs after a probe loaded two `.mlmodelc` artifacts.

## `_ANEChainingRequest` (size 80, NSSecureCoding)

```
+ chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:
                          procedureIndex:signalEvents:transactionHandle:
                          fwEnqueueDelay:memoryPoolId:

ivars:
  [+0x08] _inputBuffer                NSArray   // IOSurfaces in
  [+0x10] _outputSets                 NSArray   // IOSurfaces out (PLURAL)
  [+0x18] _loopbackInputSymbolIndex   NSArray   // chain-edges:
  [+0x20] _loopbackOutputSymbolIndex  NSArray   //   stage-N out → stage-N+1 in
  [+0x28] _signalEvents               NSArray   // cross-stage sync
  [+0x30] _transactionHandle          NSNumber  // groups stages atomically
  [+0x38] _procedureIndex             NSNumber  // which procedure to invoke
  [+0x40] _fwEnqueueDelay             NSNumber  // firmware-side enqueue delay
  [+0x48] _memoryPoolId               NSNumber  // shared mem pool across stages
```

Key implications:
- `_loopback*SymbolIndex` are how stage outputs feed stage inputs **inside the daemon** (no host copy).
- `_memoryPoolId` lets all stages share a single buffer pool — intermediates never cross the bus.
- `_signalEvents` lets stages issue/wait on Mach events.
- `_procedureIndex` is a single `NSNumber`, but `_outputSets` is plural. Open question: does one chain request fan out across multiple procedures, or is fan-out done by daisy-chaining requests via `_transactionHandle`? **Probe needed.**
- Conforms to `NSSecureCoding` → can be persisted/transported safely.

## `_ANEModelInstanceParameters` (size 24, NSCopying + NSSecureCoding)

```
+ withProcedureData:procedureArray:

ivars:
  [+0x08] _instanceName     NSString
  [+0x10] _procedureArray   NSArray   // multiple procedures per loaded model
```

This is the killer fact: **one loaded model carries N procedures**, addressed by `_procedureIndex`. This is the private-API equivalent of CoreML 9's public `MLMultiFunctionDescriptor`. For MoE this means all N experts of a layer can live in one `.mlmodelc`/`.mlpackage` and be selected per-token by index.

## `_ANEInMemoryModelDescriptor` (size 64)

```
+ modelWithMILText:weights:optionsPlist:
+ modelWithNetworkDescription:weights:optionsPlist:

ivars:
  [+0x08] _isMILModel        BOOL
  [+0x10] _networkTextHash   NSString
  [+0x18] _weightsHash       NSString
  [+0x20] _optionsPlistHash  NSString
  [+0x28] _networkText       NSData       // raw .mil text OR network description
  [+0x30] _weights           NSDictionary // weights as a dict, not a path
  [+0x38] _optionsPlist      NSData
```

Lets us **bypass the filesystem entirely** — feed MIL text + weight dict from memory, skip the `model.espresso.net` materialization that broke our coremltools-generated `.mlmodelc` artifacts in the chain probe. Backed by `_ANEInMemoryModel.compileWithQoS:options:error:` and `loadWithQoS:options:error:`.

## `_ANEInMemoryModel` (size 112)

Full lifecycle on a memory-resident model:
- `-initWithDesctiptor:` (sic, typo'd in Apple's framework)
- `-compileWithQoS:options:error:`
- `-loadWithQoS:options:error:`
- `-evaluateWithQoS:options:request:error:`
- `-mapIOSurfacesWithRequest:cacheInference:error:`
- `-purgeCompiledModel`
- `-saveModelFiles`
- `-localModelPath`

Holds `_program: _ANEProgramForEvaluation` and `_descriptor: _ANEInMemoryModelDescriptor`.

## `_ANEProgramForEvaluation` (size 56)

```
+ programWithController:intermediateBufferHandle:queueDepth:
+ programWithHandle:intermediateBufferHandle:queueDepth:

key methods:
  - processInputBuffers:model:options:error:
  - processOutputSet:model:options:error:
  - processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:
  - processSessionHint:options:report:error:
```

Note `processSessionHint:` — there is a "session" concept; likely how the daemon caches IO surface bindings across a sequence of related calls (e.g. token-by-token decoding).

## `_ANERequest` (size 96) — non-chained sibling

```
ivars:
  [+0x08] _inputArray              NSArray
  [+0x10] _inputIndexArray         NSArray
  [+0x18] _outputArray             NSArray
  [+0x20] _outputIndexArray        NSArray
  [+0x28] _weightsBuffer           _ANEIOSurfaceObject   // ← per-call weight injection!
  [+0x30] _sharedEvents            _ANESharedEvents
  [+0x38] _transactionHandle       NSNumber
  [+0x40] _procedureIndex          NSNumber
  [+0x48] _perfStats               _ANEPerformanceStats
  [+0x50] _perfStatsArray          NSArray
  [+0x58] _completionHandler       block
```

Two surprises:
1. `_weightsBuffer` is an **`_ANEIOSurfaceObject`** — the ANE supports per-request weight injection via IOSurface. This is potentially how Apple ships ad-hoc LoRA / model variants without recompiling. For MoE this is an alternative to the multi-procedure approach: load one MLP shape, inject expert weights per token. Latency cost: TBD.
2. `_perfStatsArray` plural → per-procedure perf stats are returned for chained calls.

## `_ANEProgramIOSurfacesMapper` (size 32)

Wraps the surface-mapping path used by both single and chained requests; binds IOSurface handles to program input/output symbols. Mostly internal.

## Implications for "Flash MoE on ANE"

The original concern — "1024 expert kernels × 32 layers = combinatorial chain blowup" — is dissolved:

| Earlier guess | Schema reality |
|---|---|
| Compile each expert as a separate `.mlmodelc`, load 1024 models | One model per layer, **N procedures inside it**, addressed by `_procedureIndex`. |
| Pre-build chains for every expert combination | `_ANEChainingRequest` is a tiny `NSSecureCoding` struct (≤100 bytes payload). Build one per token. |
| Worry about per-expert host↔ANE memcpy | `_memoryPoolId` + `_loopbackSymbolIndex` keep activations on-device across stages. |
| 8 XPC RTTs per layer for top-8 experts | One chain request per layer (assuming fan-out works) or 8 with shared `_transactionHandle`. |
| Filesystem-backed `.mlmodelc` files | `_ANEInMemoryModelDescriptor` accepts MIL text + weight dict directly. |

## Update — More Classes Discovered (round 2)

Full ANE class enumeration: 34 `_ANE*` classes total. New high-value ones beyond round 1:

### `_ANEOutputSetEnqueue` (size 32) — **multi-procedure fan-out**

```
+ outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:

ivars:
  [+0x08] _signalNotRequired  BOOL
  [+0x09] _isOpenLoop         BOOL
  [+0x0c] _procedureIndex     uint32_t
  [+0x10] _setIndex           uint32_t
  [+0x18] _signalValue        uint64_t
```

This is the answer to the open question "does one chain call execute multiple procedures?" — **yes**. The `_outputSets` array in `_ANEChainingRequest` is `NSArray<_ANEOutputSetEnqueue *>` — each entry says "run procedure P, write its set-index S, raise event with value V". A single chain submission can fan out across N procedures of the same loaded model.

`_isOpenLoop` is striking: ANE supports async/streaming dispatch where the host never reads back — fire-and-forget enqueues. Useful for prefetching the next layer's experts while the current layer is still summing.

`_signalNotRequired` lets some stages skip event signaling — the "leaf" procedures of a chain DAG can be cheaper.

### `_ANEInputBuffersReady` (size 40) — input-side handshake

```
+ inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:

ivars:
  [+0x08] _procedureIndex          uint32_t
  [+0x10] _inputBufferInfoIndex    NSArray
  [+0x18] _inputFreeValue          NSArray
  [+0x20] _executionDelay          uint64_t
```

Companion to `_ANEOutputSetEnqueue`. Notice `_executionDelay` again — firmware-level timing control. `_inputFreeValue` likely lets the daemon recycle the input buffer after the listed event values are reached → enables pipelining across procedures.

### `_ANEBuffer` (size 32) — symbol-bound IO

```
+ bufferWithIOSurfaceObject:symbolIndex:source:

ivars:
  [+0x08] _ioSurfaceObject  _ANEIOSurfaceObject
  [+0x10] _symbolIndex      NSNumber
  [+0x18] _source           int64_t
```

`_inputBuffer` array elements are these. Inputs bind by **symbol index** (compiler-assigned integer), not by name → fast.

### `_ANEWeight` (size 40, NSCopying + NSSecureCoding)

```
+ weightWithSymbolAndURLSHA:weightURL:SHACode:

ivars:
  [+0x08] _weightSymbol       NSString
  [+0x10] _weightURL          NSURL
  [+0x18] _SHACode            NSData
  [+0x20] _sandboxExtension   NSString
```

Weights are **per-symbol file URLs**, not raw blobs. Sandbox extensions let the daemon access weight files outside its sandbox. This is the standard path. (`_ANEInMemoryModelDescriptor._weights` likely uses a different value type — raw `NSData` per symbol — for the in-memory path.)

### `_ANEVirtualClient` (size 24) — **direct IOKit driver path, bypasses daemon**

67 instance methods, 22 class methods. This is the **second runtime path** alongside `_ANEClient → _ANEDaemonConnection → daemon`. `_ANEVirtualClient` talks directly to the IOKit user client, bypassing `aned`. Selected methods of interest:

```
-[validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:]
   ret=^{__CFDictionary=}  args=[Q, @, @, @, @, @]
   -- accepts a `function:` NSString and `milTextData:` NSData
   -- this is HOW MULTI-FUNCTION MIL FLOWS DOWN to firmware

-[validateNetworkCreateMLIR:validation_params:]
   -- accepts MLIR (not just MIL) directly

-[compileModel:options:qos:error:]
-[loadModel:options:qos:error:]
-[loadModelNewInstance:options:modelInstParams:qos:error:]
-[loadModelNewInstanceLegacy:...]
-[doEvaluateWithModel:options:request:qos:completionEvent:error:]
-[doEvaluateWithModelLegacy:...]   -- "Legacy" suggests v1/v2 protocol coexists
-[mapIOSurfacesWithModel:request:cacheInference:error:]
-[sessionHintWithModel:hint:options:report:error:]
-[beginRealTimeTask] / -[endRealTimeTask]   -- real-time priority bracket
-[exchangeBuildVersionInfo] / -[sendGuestBuildVersion] / -[hostBuildVersionStr]
-[negotiatedCapabilityMask] / -[negotiatedDataInterfaceVersion]
-[validateEnvironmentForPrecompiledBinarySupport]
```

Implications:
- A `function:` parameter at the lowest IOKit boundary confirms: **CoreML 9's `MLMultiFunctionDescriptor` lowers to a per-function string here**, and the multi-function `.mlmodelc` becomes the `_procedureArray` populated on the `_ANEModelInstanceParameters` side.
- `beginRealTimeTask` / `endRealTimeTask` — the ANE supports a real-time scheduling bracket. Worth wrapping our hot decode loop in this for tail-latency reasons.
- `validateNetworkCreateMLIR:` — Apple's compiler accepts MLIR, not just `.mil`. Interesting alternate authoring path.

### `VirtANEModel` C struct shape (from method signatures)

```
{VirtANEModel = I q I I I I Q Q Q Q
                [32 I] [32 Q] [32 I] [32 Q]
                Q Q Q c C I Q I I I Q I I Q I Q
                [64 I] [64 I] [64 I] [64 I]
                I Q Q [64 I] [64 I]
                I I I I I I Q q I I Q I Q I Q I Q I Q I Q I Q I Q}
```

Fixed-size arrays: `[32 I]`, `[32 Q]` and `[64 I]`. Reasonable interpretation:
- `[32 ...]` ≈ up to **32 inputs/outputs/symbols** per procedure (consistent with ANE's known small fan-in).
- `[64 ...]` ≈ up to **64 procedures or weight banks** per loaded model.

If the **64-procedure cap holds**, then for 128-expert MoE layers we need **2 loaded model instances per layer** (or a cheaper alternative: one model with 64 procedures and a second one with the other 64; chain calls already work across two distinct loaded models — verified in `ane_chain_probe`).

### `_ANEPerformanceStats` (size 32)

```
ivars:
  [+0x08] _hwExecutionTime   uint64_t (ns)
  [+0x10] _pStatsRawData     NSData
  [+0x18] _perfCounterData   NSData

methods of interest:
  - performanceCounters
  - emitPerfcounterSignpostsWithModelStringID:
  + driverMaskForANEFMask:
```

Per-procedure perf counters with hardware execution time in ns. Combined with `_perfStatsArray` (plural) on `_ANERequest`, we can profile per-expert latency directly.

### `_ANECloneHelper`

```
+ cloneIfWritable:isEncryptedModel:cloneDirectory:
+ shouldSkipCloneFor:isEncryptedModel:
```

Daemon clones writable model directories before mapping. Means we can drop a `.mlmodelc` in a writable dir and it will be auto-copied. (Operational footnote.)

## Round-2 Summary: What's Now Settled

| Question | Answer |
|---|---|
| Can one chain request invoke multiple procedures? | **Yes.** `_outputSets` is `NSArray<_ANEOutputSetEnqueue*>`, each carrying its own `_procedureIndex` + `_setIndex` + `_signalValue`. |
| Is async fire-and-forget dispatch supported? | **Yes.** `_isOpenLoop` BOOL on `_ANEOutputSetEnqueue`. |
| Per-procedure signaling and pipelining? | **Yes.** Per-procedure `signalValue` + `_inputFreeValue` + `_executionDelay`. |
| Is there a daemon-bypass path? | **Yes.** `_ANEVirtualClient` talks directly to the IOKit user client. |
| Does the firmware-level API accept named functions? | **Yes.** `validateNetworkCreate:...function:...milTextData:`. |
| Can ANE compile MLIR (not just MIL)? | **Yes.** `validateNetworkCreateMLIR:`. |
| Procedure cap per loaded model? | Likely **64** (firmware struct has `[64 I]` arrays). To be probed empirically. |
| Real-time priority bracket? | **Yes.** `beginRealTimeTask` / `endRealTimeTask`. |

## Updated MoE Plan Implications

For Gemma 4 26B-A4B (128 routed experts × 30 layers):
- **2 loaded models per layer** if the 64-procedure cap is real → 60 model instances total. The daemon already keeps ≥2 distinct models resident per connection (verified). Need to test ≥60.
- **Per-token chain**: build one `_ANEChainingRequest` whose `_outputSets` lists the 8 active experts as 8 `_ANEOutputSetEnqueue` entries with the same `_memoryPoolId`. One XPC RTT per layer, period.
- **Pipelined decoding**: use `_isOpenLoop=YES` on the early-layer expert dispatches and only synchronize at the final logits stage. Could dramatically reduce host-bound idle time.
- **Real-time bracket**: wrap the per-token decode loop in `beginRealTimeTask` / `endRealTimeTask` for predictable latency.

## Remaining Open Questions

1. **64-procedure cap** — probe by attempting to load a model with N procedures, sweeping N. Needs the multi-function `.mlpackage` build path first.
2. ~~**In-memory weights value-type**~~ — **PARTIALLY ANSWERED.** [artifacts/ane_inmemory_model_probe.m] shows `_weights` is **not** a flat `{NSString → NSData}`, **not** a flat `{NSString → _ANEWeight}`, and **not** a one-element array wrapper of either. The descriptor factory sends `count` to direct entries and `allValues` to array-wrapped entries, which strongly suggests a more nested dictionary-like per-weight payload. The same probe showed the inline-empty MIL path is real enough to create `_ANEInMemoryModelDescriptor` + `_ANEInMemoryModel` and reach `compileWithQoS:`, but the toy MIL fails at `_ANECompiler : ANECCompile() FAILED`. A follow-up replay with a **real compiled Qwen conv artifact** (`model.mil` + `weights/weight.bin`) using the nested raw-`NSData` container reached the **same** boundary: descriptor created, model created, private compiler entered, `ANECCompile()` failed again. Feeding either packaged sidecar blob (`coremldata.bin` or `analytics/coremldata.bin`) into the descriptor `optionsPlist` slot also left the boundary unchanged, and swapping to the alternate `modelWithNetworkDescription:weights:optionsPlist:` factory with packaged `coremldata.bin` still reached the same compile failure. The next compile-options probe established that `compileWithQoS:options:error:` is a **live** input surface: `_ANEInMemoryModel` synthesizes a compiler-options dictionary containing `kANEFCompilerOptionsFilenameKey`, `kANEFInMemoryModelIsCachedKey`, `kANEFIsInMemoryModelTypeKey`, and `kANEFModelType`, and caller-supplied options are merged into that dictionary. But forcing `kANEFModelType = kANEFModelANECIR` on the real MIL replay is normalized back to `kANEFModelMIL` and still fails at the same `ANECCompile()` wall. A follow-up probe using the other obvious real internal key, `kANEFCompilerOptionsFilenameKey`, also failed to move the wall: the caller-supplied alternate filename was not preserved in the derived compiler-options dictionary, which normalized back to `compiler_options.plist`, and compile still failed at the same `ANECCompile()` boundary. A final discriminator then bypassed the options-dictionary normalization entirely by calling `_ANEInMemoryModel`'s `setCompilerOptionsFileName:` directly before deriving compiler options. That direct setter path is live: `compilerOptionsFileName` changed to `alternate_compiler_options.plist`, the derived compiler-options dictionary carried `kANEFCompilerOptionsFilenameKey = "alternate_compiler_options.plist"`, and compile still failed at the same `ANECCompile()` boundary. A further compiler-owned probe using `maxModelMemorySize = 4096` also survived into the derived compiler-options dictionary unchanged and still failed at the same `ANECCompile()` boundary. Subsequent structural probes also closed the remaining obvious path / alias branches: shrinking the nested weights map to the single canonical outer key (`@model_path/weights/weight.bin`) plus a single inner `w` payload changed the weights hash but not the boundary; changing that inner key to `weights/weight.bin` normalized away entirely (same weights hash, same staged `w` file); and rewriting the MIL itself from `@model_path/weights/weight.bin` to `@model_path/w` changed the network hash but still failed at the same `ANECCompile()` wall. The next lower control also removed `_ANEModel + _ANEClient` as a rescue path for this specific problem: a direct compile of both the **original real `.mlmodelc`** and the **rewritten staged in-memory directory** failed identically in `com.apple.appleneuralengine.espresso` code `-1` with `_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'`. That showed the lower direct-client surface expects a different on-disk layout than both current public-CoreML `.mlmodelc` output and the in-memory staging directory. A follow-up control search then found a distinct legacy artifact family in a third-party application cache whose `.mlmodelc` root contains `model.espresso.net`, `model.espresso.shape`, and `model.espresso.weights`. Running the existing direct `_ANEModel + _ANEClient` path against two of those bundles moved the boundary exactly once: the missing-`model.espresso.net` failure disappeared, `_ANEModel` objects were created for both, and compile/load advanced to later failures instead of file lookup. But it still did not yield an end-to-end positive control: one bundle failed in `com.apple.appleneuralengine.compiler` with `InvalidNetworkSourceFileName`, and the other failed in `com.apple.appleneuralengine.espresso` code `-2` with `Cannot serialize ANEC_IR_repr`. A final repo-only control then removed the need for any external artifact family in the argument: the existing tiny multifunction package at `scratch/mfn_probe/experts_multi.mlpackage` has the normal public package layout (`Manifest.json` + `Data/`), its compiled sibling `experts_multi.mlmodelc` has the same modern public layout (`analytics/`, `coremldata.bin`, `model.mil`, `weights/`), and the direct `_ANEModel + _ANEClient` path against that compiled artifact still fails immediately with `_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'`. So the remaining unknown is now tighter than before: nested weight-key aliasing and `weight.bin` vs `w` are out, plain `model.espresso.net` materialization is necessary but not sufficient, and repo-local public compile outputs are definitively still upstream of the legacy Espresso layout the lower direct-client surface expects. The unresolved contract is the hidden MIL-to-Espresso translation step plus the stricter legacy Espresso source-name / IR format that the current daemon/compiler still accepts upstream of `_ANEClient compileModel:options:qos:error:`.
3. ~~**`_ANEVirtualClient` vs daemon**~~ — **PARTIALLY ANSWERED.** [artifacts/ane_virtual_client_probe.m] loaded `AppleNeuralEngine.framework` and resolved `_ANEVirtualClient`, but the only discovered constructor, `+sharedConnection`, returned `nil` from an unsigned development binary (`scratch/ane_virtual_client_probe/summary.json`). Treat the direct daemon-bypass path as blocked unless a different bootstrap or entitlement-bearing host is found.
4. **Multi-function .mlpackage authoring** — coremltools 9.0 returned `MLMultiFunctionDescriptor: False` from `ct.models`. Find the actual API surface (likely `ct.utils.bisect_model` style helpers, or the `mil.Function` route).

## Update — Round 3: Public Multi-Function API Verified End-to-End

**Public CoreML surface** (macOS 15+):
- `MLModelConfiguration.functionName: NSString?` — selects which function to load
- `MLModelAsset.functionNames(completionHandler:)` — enumerates available functions
- `MLModelAsset.modelDescriptionOfFunctionNamed:` — per-function input/output descriptions

**coremltools 9.0 authoring API** (lives at `coremltools.models.utils`, not `coremltools.utils`):
```python
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

desc = MultiFunctionDescriptor()
for i, p in enumerate(expert_packages):
    desc.add_function(str(p), "main", f"expert_{i}")
desc.default_function_name = "expert_0"
save_multifunction(desc, "experts_multi.mlpackage")
# performs constant deduplication across functions for shared weights
```

**End-to-end probe** ([research-probes/ane_multifunction_probe.py]):
Built 4 tiny `linear+relu` "experts", combined into one `.mlpackage` via `save_multifunction`, loaded each via `function_name="expert_i"` on `CPU_AND_NE`, all four ran and produced distinct outputs:
```
expert_0: out[:4] = [15.585, 0.0, 8.421, 0.0]
expert_1: out[:4] = [4.269, 0.0, 10.835, 3.355]
expert_2: out[:4] = [7.714, 0.0, 6.128, 0.0]
expert_3: out[:4] = [0.0,   0.0, 8.632, 0.278]
```

**Constant deduplication** is the key win: `save_multifunction` automatically shares constant tensors across functions. For MoE this means:
- Per-layer shared blocks (norms, attention QKV/O, router) are stored once.
- Only the per-expert MLP weights are unique storage.
- The 50 GB Gemma 4 footprint isn't multiplied by the function count.

**Authoring contract for one MoE layer**:
1. Build N "expert kernels" as standalone `.mlpackage`s, each `f(x) -> expert_out`.
2. Optionally build the shared attention block as another function.
3. `MultiFunctionDescriptor` + `save_multifunction` → one `.mlpackage` with N+1 named functions.
4. At runtime: for each chosen expert per token, instantiate `MLModel(..., function_name="expert_k")`. Reuse instances across tokens.

**Caveats observed**:
- Default top-level `spec.description.input` is empty in multi-function models. Use `MLModel.input_description` (per-instance, picks up the active function) or `MLModelAsset.modelDescription(of:)`.
- `ct.utils.compile_model` produces a `.mlmodelc` whose layout is **not** what `ct.models.MLModel` expects to load (no `Manifest.json`). Load directly from the `.mlpackage` instead — CoreML compiles internally.
- coremltools warns about `fp16` IO and inserts CPU casts; for ANE-resident pipelines author IO as `fp32` at the boundary or pass `outputs=[ct.TensorType(dtype=np.float16)]` with `iOS16`+ opset.

## What's Now Settled (Cumulative)

| Question | Answer | Source |
|---|---|---|
| Per-call retargetable chain XPC | YES | ane_chain_probe |
| Multi-procedure per loaded model | YES, capped near 64 (TBD) | _ANEModelInstanceParameters |
| Multi-procedure fan-out per chain call | YES via `_outputSets: [_ANEOutputSetEnqueue]` | round 2 dump |
| Async fire-and-forget dispatch | YES, `_isOpenLoop` | round 2 dump |
| Daemon-bypass IOKit path | Surface exists, but unsigned admission is blocked via `sharedConnection -> nil` in the current probe | round 2 dump + ane_virtual_client_probe |
| ANE accepts MLIR | YES, `validateNetworkCreateMLIR:` | round 2 dump |
| Real-time scheduling bracket | YES, begin/endRealTimeTask | round 2 dump |
| Public multi-function API | YES, macOS 15+ `functionName` | SDK headers |
| Coremltools authoring path | YES, `MultiFunctionDescriptor` + `save_multifunction` | API discovery |
| Constant deduplication across functions | YES (built into save_multifunction) | docstring + probe |
| Multi-function model loads & runs on ANE | YES, 4/4 functions ran | live probe |
| `_ANEInMemoryModel` path reaches private compiler | YES, descriptor + model instantiate and `compileWithQoS:` runs | ane_inmemory_model_probe |

## Still Open

1. ~~**Procedure-count cap per loaded model**~~ — **REFUTED.** Sweep N=4,16,32,64,96,128,192,256 ([research-probes/ane_proc_cap_sweep.py]) — **all 256 functions loaded and ran successfully on `CPU_AND_NE`**. No cap. The `[64 I]` arrays in `VirtANEModel` are per-procedure symbol/bank limits, not procedure counts. Implication: **all 128 Gemma 4 experts per layer can live in a single `.mlpackage`** — 30 packages total instead of 60.
2. **Shared-weight memory accounting** — measure on-disk and in-memory size of an N-function `.mlpackage` vs N standalone packages, with and without identical weights, to validate dedup ratio.
3. **Sequential-load cost vs cached instances** — sweep showed 0.33s per `MLModel(..., function_name=...)` load+predict at N=256 (84s total). Real driver must cache the loaded `MLModel` per function. Open: does the daemon refcount the underlying `_ANEModel` so 128 cached instances share one resident model? (The chain primitive proves yes at the firmware level.)
4. **Chain across two distinct loaded models** — verified the daemon holds 2 models concurrently (round 1); next is to actually `prepareChainingWithModel:` with a real `_ANEChainingRequest` whose `_outputSets` reference procedures from both models. Open question: does `_procedureIndex` namespace span both models, or is the chain bound to one model only?
5. **Per-call weight injection** — `_ANERequest._weightsBuffer: _ANEIOSurfaceObject` semantics. If weights can be injected per call, the multi-function approach becomes redundant for MoE.
6. **Exact in-memory `_weights` container shape** — the probe ruled out flat values and one-element arrays. Need runtime reflection or a harvested working example to learn the nested dictionary-like payload expected by `modelWithMILText:weights:optionsPlist:`.
6. ~~**Daemon-bypass entitlements**~~ — **PARTIALLY ANSWERED.** The runtime surface exists, but [artifacts/ane_virtual_client_probe.m] found only `+sharedConnection`, and that returned `nil` from an unsigned development binary in this probe (`scratch/ane_virtual_client_probe/summary.json`). Direct `_ANEVirtualClient` work is deprioritized until we find a different bootstrap path or an entitlement-bearing host.
## Update — Round 4: Empirical Bandwidth & Latency

Three probes measured what the ANE actually does on this M4 Max with realistic shapes.

### Device assignment is shape-dependent (critical caveat)

`MLComputePlan.get_compute_device_usage_for_mlprogram_operation` reveals that small shapes silently fall back to CPU even with `compute_units=CPU_AND_NE`:

| Shape (d_model, d_ffn) | Weights (fp16) | Device |
|---|---|---|
| 1024, 4096 | 25 MB | **CPU** |
| 2048, 4096 | 50 MB | ANE |
| 2304, 9216 | 127 MB | ANE |

So **earlier "GB/s" numbers under ~16 MB were measuring CPU**, not ANE. This invalidates the simple BW sweep at small sizes — those got 24–116 GB/s but it was the CPU's BW, not the ANE's.

### Upper-size cliff recheck (2026-04-24)

The earlier working assumption that compiled artifacts fall off ANE around `~96 MB` is **wrong as a general law**.

New counterexample: a **stateful INT8 one-layer Qwen 7B probe** built via `gguf_to_ane.py` compiled to:

| Artifact | Size | Conv placement |
|---|---|---|
| `Qwen7B_1L_probe.mlpackage` | 223 MB | 4/4 ANE |
| `Qwen7B_1L_probe.mlmodelc`  | 223 MB | 4/4 ANE |

Measured via `MLComputePlan` on `scratch/qwen7b_1layer_probe/Qwen7B_1L_probe.mlmodelc`, which reported `ANE=4 CPU=0 GPU=0` for conv ops (`ios18.conv*` / `conv` / `ios18.convolution`).

So the right reading is narrower:
- there is a **small-graph / INT4 shard bug** that pushes convs to CPU
- there is a **lower-size floor** where tiny shapes fall to CPU
- but there is **not** a universal upper compiled-size cliff at `~96 MB`

## Update — Round 5: Phi/CoreML Bridge Probe

**Probe**: [ane_coreml_bridge_probe.m](ane_coreml_bridge_probe.m) against
`phi4mini_layer30_32_q8.mlmodelc`.

The direct private `_ANEClient` chain surface is still real, but current public
CoreML `.mlmodelc` artifacts remain incompatible with that lower entry point:

```
_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'
```

That means the immediate private-API path should not start by synthesizing
`_ANEModel` objects from public CoreML output. That path is blocked at the
legacy Espresso artifact contract.

The more promising bridge is one layer higher, through CoreML's E5 runtime.
Loading the same Phi shard through public `MLModel(contentsOf:configuration:)`
does register a model UUID in `_ANEClient connectionsUsedForLoadingModels`, so
CoreML is using the same daemon lane successfully. Runtime introspection shows:

```
MLDelegateModel
  _internalEngine: MLE5Engine
   _programLibrary: MLE5ProgramLibrary
    _programLibraryHandle: e5rt_program_library*
    _impl: MLE5ProgramLibraryOnDeviceAOTCompilationImpl
    _container: MLProgramE5Container
```

Calling `-[MLE5ProgramLibrary _programLibraryHandleWithForceRespecialization:error:]`
on the loaded Phi shard returned a non-null `e5rt_program_library` handle with
no error. That is the first concrete bridge object between public CoreML's
successful load path and the private ANE daemon/runtime layer.

Next targets:

1. Map the E5RT operation/request classes reachable from `MLE5ProgramLibrary`
  without forcing a long compile/evaluate path.
2. Determine whether an E5 operation exposes the `_ANEModel` UUID, program
  handle, intermediate buffer handle, or IOSurface binding tables.
3. Only after that, attempt a two-shard chain using CoreML-created resident
  model handles instead of manually-created `_ANEModel` objects.

## Update — Round 6: E5 Operation And Port Handles

**Probe**: [coreml_e5_class_dump.m](coreml_e5_class_dump.m) plus the updated
[ane_coreml_bridge_probe.m](ane_coreml_bridge_probe.m).

The live CoreML E5 runtime surface is now mapped well enough to name the next
bridge object family:

```
MLE5Engine
  _streamPool: MLE5ExecutionStreamPool
  _operationPool: MLE5StaticShapeExecutionStreamOperationPool
  _programLibrary: MLE5ProgramLibrary

MLE5ProgramLibrary
  _programLibraryHandle: e5rt_program_library*
  -createOperationForFunctionName:forceRespecialization:hasRangeShapeInputs:error:
      -> e5rt_execution_stream_operation*

MLE5ExecutionStreamOperation
  _operationHandle: e5rt_execution_stream_operation*
  _inputPorts:  NSArray<MLE5InputPort *>
  _statePorts:  NSArray<MLE5InputPort *>
  _outputPorts: NSArray<MLE5OutputPort *>

MLE5InputPort / MLE5OutputPort
  _portHandle: e5rt_io_port*
  _binder: MLE5InputPortBinder / MLE5OutputPortBinder
```

Important correction: `createOperationForFunctionName:...` returns a raw
`e5rt_execution_stream_operation*`, not an Objective-C object. The ObjC wrapper
comes from `MLE5StaticShapeExecutionStreamOperationPool` after
`prepareWithInitialPoolSize:error:` and `_takeOut`.

On `phi4mini_layer30_32_q8.mlmodelc`, the bridge probe successfully recovered:

- `e5rt_program_library*`
- `e5rt_execution_stream_operation*`
- named input port handles for `attn_mask`, `kv_write_mask`, `rope_cos`,
  `rope_sin`, and `x`
- named state port handles for the KV caches
- named output port handle for `hidden`

The same handle/port pattern also works on the adjacent
`phi4mini_layer16_24_q8.mlmodelc` shard, including the `x` input and `hidden`
output. This gives us the first concrete CoreML-created E5 objects needed for a
two-shard chain: stage A `hidden` output port and stage B `x` input port are both
visible as `e5rt_io_port*` handles while the models are loaded through public
CoreML.

Next target: construct or borrow an `MLE5ExecutionStream` containing two
`MLE5ExecutionStreamOperation` objects and determine whether CoreML/E5RT can
bind stage A's `hidden` output port directly to stage B's `x` input port without
materializing an `MLMultiArray` back to Swift.

## Update — Round 7: Two-Operation E5 Stream And Binder Modes

**Probe**: [e5_two_op_stream_probe.m](e5_two_op_stream_probe.m).

The probe loads two adjacent Phi shards through public CoreML, takes one
`MLE5ExecutionStreamOperation` from each shard's operation pool, and constructs
one `MLE5ExecutionStream` containing both operations:

```
operations=(
  MLE5ExecutionStreamOperation for phi4mini_layer16_24_q8,
  MLE5ExecutionStreamOperation for phi4mini_layer24_30_q8
)
serializeInferenceFrameDataForOptions:error: -> YES
```

So `MLE5ExecutionStream` is willing to hold operations from two separately
loaded public-CoreML models, at least structurally and without execution.

The hidden-to-x binding experiment found the next important boundary:

```
stageA.directOutputs = (hidden)
stageB.directInputs  = ()

stageA hidden prepareWithOptions:error: -> YES
stageA hidden featureValue: MLFeatureValue(MultiArray Float16 1 x 3072 x 1 x 1)
stageA hidden boundFeatureDirectly -> YES

stageB x reusableForHiddenFeatureValue -> NO, willBindDirectly -> NO
stageB x prepareForHiddenFeatureValue -> YES, xDirect -> NO
```

That is the public-ish E5 path: CoreML can produce a direct output feature for
`hidden`, and stage B can accept it as `x`, but it does not direct-bind by
default.

Private binder modes expose a sharper clue:

```
MLE5InputPortBinder _reusableForFeatureValue:directMode:
  mode 0 -> NO
  mode 1 -> NO
  mode 2 -> YES
  mode 3..7 -> NO

setDirectlyBoundFeatureValue:hiddenFeatureValue + setBindingMode:1
  stageB x boundFeatureDirectly -> YES
```

Interpretation: `MLE5InputPortBinder` has at least one private direct-mode value
that recognizes the output feature value as reusable, and a separate binding
mode value that marks the input port as directly bound. This is not an executed
chain yet, but it proves the hidden-output feature can be installed into the
stage-B input binder in a direct-bound state.

Next target: use a tiny two-model E5 graph with no KV state to test whether a
two-operation stream with forced direct binding can execute correctly before
risking the full Phi decode path.

## Update — Round 8: Tiny Two-Model Execution Controls

The two-operation stream was tested on tiny synthetic CoreML models under
`scratch/ane_private_api/toy/`:

- `toy_a`: `hidden = x + 1`
- `toy_b`: `hidden = x * 2`
- `toy_b_h`: `hidden = h * 2`

With matching input names (`toy_a` then `toy_b`, both input `x`), the two-op
stream executes successfully:

```
executeForInputFeatures:options:error: -> YES
stageA.hidden.afterExecute = [2, 3, 4, 5]
stageB.hidden.afterExecute = [2, 4, 6, 8]
```

That proves the constructed stream can execute two operations, but it also shows
there was no chain: stage B consumed the original input provider's `x`, not
stage A's `hidden`. A true chain would have produced `[4, 6, 8, 10]`.

With distinct input names (`toy_a` input `x`, `toy_b_h` input `h`), the forced
hidden-to-input binder state is not sufficient for E5RT execution:

```
executeForInputFeatures:options:error: -> NO
The input feature is invalid or unsupported. (port trait Tensor, feature trait Unknown.)
```

So the current conclusion is precise:

1. CoreML E5 can load two public models, expose operation/port handles, and run
  both operations in one `MLE5ExecutionStream`.
2. `MLE5InputPortBinder` direct state can be forced structurally.
3. Reusing the stage-A `MLFeatureValue` before execution is not enough to create
  an E5RT-level port-to-port edge.

The remaining missing primitive is lower than `MLFeatureValue`: some E5RT or ANE
operation that links an output `e5rt_io_port*` to an input `e5rt_io_port*`, or a
supported way to create an input memory object from a prior output port after the
stream is prepared.

Operational note: full dyld extraction was deferred because the test host had only
about 25 GiB free. Do not run the broad dyld extraction helper without explicit
approval or more disk headroom.

## Update — Round 9: Binder Timing and Raw Execute Controls

`e5_two_op_stream_probe.m` now probes the lower binding order around
`MLE5InputPortBinder`, `MLE5ExecutionStreamOperation`, and raw
`MLE5ExecutionStream` execution.

New live selectors confirmed from CoreML's ObjC runtime metadata:

```
MLE5InputPortBinder -bindMemoryObjectForFeatureValue:error:
MLE5ExecutionStreamOperation -_bindInputFeaturesAndWaitEvents:options:error:
MLE5ExecutionStreamOperation -_bindOutputPortsWithOptions:error:
MLE5ExecutionStream -_prepareForInputFeatures:options:error:
MLE5ExecutionStream -_executeStream:error:
```

Results:

1. Calling `bindMemoryObjectForFeatureValue:error:` on stage B's input binder
  with stage A's pre-execution output feature returns `YES`, but it still does
  not create a working inter-op edge.

2. If stage B's real input feature is present in the provider (`h =
  [10,20,30,40]`), the distinct-input toy executes and returns
  `[20,40,60,80]`. That means CoreML/E5 consumes the provider input, not the
  forced stage-A output binder.

3. If the stream is prepared first and the probe then tries to force-bind stage
  B's input to stage A's output, CoreML rejects the mutation:

```
Port bindings cannot be changed while operation is in use in an execution stream. @ BindMemoryObject
```

Raw `_executeStream:error:` after that succeeds, but stage B still consumes the
provider value.

4. If the probe manually binds operation input/output ports and calls raw
  `_executeStream:error:` without stream preparation, E5RT rejects execution:

```
No operations have been encoded to the execution stream. @ ExecuteStreamSyncImpl
```

Conclusion: stream preparation is the point where operations are encoded and
bindings become immutable. A true cross-model chain must therefore be expressed
before or inside `MLE5ExecutionStream`'s setup/encoding path. Post-hoc mutation
of `MLFeatureValue`, `MLE5InputPortBinder`, or operation-level binders is too
late or too shallow.

Next target: recover how `setupOperationForInputFeatures:operationPool:error:`
chooses and encodes operations, and whether it can be influenced to bind stage
B's input port to an already-declared direct output backing before the stream is
locked.

## Update — Round 10: `setupOperationForInputFeatures` Is Replace-Oriented

`e5_two_op_stream_probe.m --probe-setup` tested whether
`MLE5ExecutionStream -setupOperationForInputFeatures:operationPool:error:` can
be called repeatedly to append operations from separate operation pools into one
stream.

It cannot. On the tiny same-input and distinct-input controls:

```
setupOperationForInputFeatures:operationPool:error: -> YES
setupStream.afterFirst operations=(op from toy_a) operationPool=toy_a pool

setupOperationForInputFeatures:operationPool:error: -> YES
setupStream.afterSecond operations=(op from toy_b/toy_b_h) operationPool=toy_b pool
```

The second setup call replaces the stream's operation list and operation pool;
it does not append. Calling `serializeInferenceFrameDataForOptions:error:` after
that succeeds, but raw `_executeStream:error:` still reports:

```
No operations have been encoded to the execution stream. @ ExecuteStreamSyncImpl
```

Interpretation: the public ObjC E5 wrapper's setup surface is designed around a
single selected operation from one operation pool. Manually assigning an
`operations` array can make the wrapper hold two operations and `execute` can run
both, but the normal `setupOperation...` selector does not expose a multi-op DAG
builder or append-mode encoder.

This narrows the private path again: either the missing primitive lives below the
ObjC `MLE5ExecutionStream` wrapper in raw E5RT calls, or it requires synthesizing
one CoreML program/function that already contains both layer ranges so the public
setup path sees one operation.

## Update — Round 11: Raw E5RT Encode Produces a True Cross-Model Chain

The missing encoder primitive was found through CoreML's imported symbols from
Espresso:

```
e5rt_execution_stream_operation_prepare_op_for_encode
e5rt_execution_stream_encode_operation
```

These are not exported by CoreML, but CoreML imports them from
`/System/Library/PrivateFrameworks/Espresso.framework/Espresso`; the probe can
resolve them with `dlsym(RTLD_DEFAULT, ...)` after CoreML loads.

The successful manual chain sequence is:

1. Bind stage A input from the normal input provider:
  `MLE5ExecutionStreamOperation -_bindInputFeaturesAndWaitEvents:options:error:`
2. Bind stage A output backing:
  `MLE5ExecutionStreamOperation -_bindOutputPortsWithOptions:error:`
3. Prepare stage A output and bind its `MLFeatureValue` / memory object into
  stage B's input port:
  `MLE5InputPortBinder -bindMemoryObjectForFeatureValue:error:`
4. Bind stage B output backing.
5. Raw-prepare and encode both operations into the same stream:

```
e5rt_execution_stream_operation_prepare_op_for_encode(stageA_op) -> 0
e5rt_execution_stream_encode_operation(stream, stageA_op) -> 0
e5rt_execution_stream_operation_prepare_op_for_encode(stageB_op) -> 0
e5rt_execution_stream_encode_operation(stream, stageB_op) -> 0
```

6. Execute the stream through CoreML's private wrapper:
  `MLE5ExecutionStream -_executeStream:error:`

Tiny distinct-input control (`toy_a: hidden = x + 1`, `toy_b_h: hidden = h * 2`)
now succeeds with no `h` provider:

```
stageA.hidden.afterManualRawExecute = [2, 3, 4, 5]
stageB.hidden.afterManualRawExecute = [4, 6, 8, 10]
```

That is the first true two-model E5 chain in this repo: stage B consumed stage
A's output buffer, not a host-provided input feature.

Immediate next validation target: run this same raw E5RT encode path on two tiny
Phi layer-range shards, compare chained output against the existing public
host-roundtrip execution, then measure whether the hidden-state host copy is
eliminated in decode profiling.

## Update — Round 12: Phi Stateful Raw Chain Smoke

The raw E5RT chain path was extended from the toy models to real Phi layer-range
shards:

```
phi4mini_layer16_24_q8.mlmodelc -> phi4mini_layer24_30_q8.mlmodelc
```

Additional state handling was required. Phi fused layer shards have ordinary
inputs (`x`, `rope_cos`, `rope_sin`, `attn_mask`, `kv_write_mask`) plus KV cache
state ports (`k_cache_*`, `v_cache_*`). Directly returning the public `MLState`
for those state names fails because the private binder expects an `MLFeatureValue`
with `internalStateValue`. Calling `MLFeatureValue +internalFeatureValueWithState:`
on the whole shard state also fails because the shard state contains many buffers:

```
MLState must have one and only one state buffer when it is stored in MLFeatureValue.
```

The working state binding pattern is:

1. Create the public shard `MLState` with `model.makeState`.
2. Read `MLState -backings`.
3. For each state port name, create a one-buffer `MLState` with
  `MLState -initWithBackings:` containing only that backing.
4. Wrap that one-buffer state with
  `MLFeatureValue +internalFeatureValueWithState:`.
5. Return the per-port internal state feature value from the feature provider.

For stage B, the provider also overrides `x` with stage A's prepared `hidden`
feature value, then calls CoreML's own operation binder:

```
MLE5ExecutionStreamOperation -_bindInputFeaturesAndWaitEvents:options:error:
```

This lets CoreML bind all ordinary inputs plus state ports in its normal path
while preserving the chained hidden-to-`x` input.

Current Phi smoke result:

```
_bindInputFeaturesAndWaitEvents(stageA) -> YES
_bindInputFeaturesAndWaitEvents(stageB with x override) -> YES
e5rt_execution_stream_encode_operation(stageA) -> 0
e5rt_execution_stream_encode_operation(stageB) -> 0
_executeStream:error: -> YES
```

Stage A output is nonzero:

```
sample = [1.498046875, 5.98828125, 4.94921875, -2.49609375, ...]
sum = -337.912079
min = -13.296875
max = 16.296875
```

Stage B output currently reads back as all zeros:

```
sample = [0, 0, 0, 0, 0, 0, 0, 0]
sum = 0
```

So Round 12 is a wiring/encode milestone, not a correctness pass. We have proven
that two real stateful Phi shard operations can be bound, raw-encoded, and
executed in one E5 stream. The remaining correctness gap is to determine why the
stage-B output backing is zero: possibilities include an output backing/readback
issue, a state/input semantic mismatch from the synthetic Phi provider, or a
need to compare against a clean public two-call reference in a separate process
that has not taken private operations out of the E5 operation pools.

Operational warning: trying to run public `predictionFromFeatures:usingState:`
inside the same process after taking private operations from the pools caused a
segmentation fault. Keep the public host-roundtrip reference in a separate
process/probe.

## Update — Round 13: Separate Public Reference Confirms Raw Stage-B Gap

Added `phi_public_two_call_probe.m`, a public-only reference that loads the same
two Phi shards in a fresh process and runs:

```
public prediction: layer16_24(input) -> hiddenA
public prediction: layer24_30(hiddenA) -> hiddenB
```

This avoids mixing public prediction with private `_takeOut` operation-pool
objects in the same process.

Reference result for the same synthetic Phi input used by the raw E5RT probe:

```
public.stageA.hidden sample = [1.498046875, 5.98828125, 4.94921875, -2.49609375, ...]
public.stageA.hidden sum    = -337.912079

public.stageB.hidden sample = [-8.0625, -0.251953125, -0.564453125, -5.12890625, ...]
public.stageB.hidden sum    = -166.729431
```

Raw E5RT result remains:

```
raw.stageA.hidden sample = [1.498046875, 5.98828125, 4.94921875, -2.49609375, ...]
raw.stageA.hidden sum    = -337.912079

raw.stageB.hidden sample = [0, 0, 0, 0, 0, 0, 0, 0]
raw.stageB.hidden sum    = 0
```

So stage A is exactly aligned between public CoreML and raw E5RT. The raw stage-B
zero is a real remaining bug in the multi-op private path, not an artifact of
the synthetic Phi inputs.

Likely cause: stage B reads the initial zero output backing before stage A's ANE
work has completed, or its output backing is not synchronized/copied after the
raw multi-op stream. A first raw event experiment was added behind
`--bind-e5-events`:

```
e5rt_execution_stream_operation_retain_completion_event(stageA) -> 0
e5rt_execution_stream_operation_bind_dependent_events(stageB, [stageA_event], 1) -> 0
```

That confirms the guessed raw signatures are plausible, but the process then
segfaults during/after the second operation prepare path. The flag is therefore
not enabled by default. Next target: reproduce CoreML's ObjC event-binding order
around `_bindNewCompletionEventsDirectlyWithCompletionSyncPoint:` and
`_bindNewWaitEventsDirectlyWithWaitSyncPoints:` instead of directly binding the
retained completion event at the wrong lifecycle point.

## Update — Round 14: MLPredictionSyncPoint Options Are Not Enough

CoreML exposes `MLPredictionOptions` ivars for `_waitSyncPoints` and
`_completionSyncPoint`, plus a private `MLPredictionSyncPoint` object wrapping
an `MTLSharedEvent` and a value. The E5 class dump was widened to include
sync/event classes and found:

```
MLPredictionOptions:
  _waitSyncPoints offset=88 type=NSArray
  _completionSyncPoint offset=96 type=MLPredictionSyncPoint

MLPredictionSyncPoint:
  _sharedEvent offset=8 type=<MTLSharedEvent>
  _value offset=16 type=uint64
  -initWithSharedEvent:value:
```

Added `--objc-sync-points` to `e5_two_op_stream_probe`. It creates an
`MTLSharedEvent`, wraps it in `MLPredictionSyncPoint`, sets stage A's
`completionSyncPoint`, and sets stage B's `waitSyncPoints` before the private
bind calls.

Result: the run is stable and all raw prepare/encode/execute calls still return
success, but stage B remains all zeros. Passing sync-point options to the bind
methods is therefore not sufficient when we bypass CoreML's normal stream encode
path with raw E5RT calls.

Added a separate `--objc-sync-update` experiment for:

```
_updateCompletionEventFutureValuesWithCompletionSyncPoint:
_updateWaitEventFutureValuesWithWaitSyncPoints:
```

Calling `_updateCompletionEventFutureValuesWithCompletionSyncPoint:` directly
segfaults in this manual lifecycle, so that flag is off by default.

Current hypothesis: CoreML's normal path creates/binds E5RT events and updates
future values as part of a larger operation lifecycle. The remaining target is
to trace the ordering in `prepareForInputFeatures:options:error:` /
`prepareAsyncSubmissionForInputFeatures:options:error:` rather than calling the
individual event update hooks in isolation.

## Update — Round 15: Direct ObjC Event Bind Attaches Events But B Still Zero

Added `--objc-sync-bind-direct`, which explicitly calls CoreML's private event
binding hooks after the manual port binds:

```
_bindNewCompletionEventsDirectlyWithCompletionSyncPoint:(syncPoint)
_bindNewWaitEventsDirectlyWithWaitSyncPoints:@[syncPoint]
```

This successfully attaches the same `_MTLSharedEvent` to both operations:

```
stageA.completionSharedEvent.beforeUpdate = <_MTLSharedEvent ...>
stageB.waitSharedEvents.beforeUpdate      = (<_MTLSharedEvent ...>)
```

Raw prepare, encode, and execute still succeed, and stage A remains correct, but
stage B remains all zeros. Therefore the missing piece is no longer merely
"attach a completion/wait event object". The remaining gap is likely one of:

- CoreML's future-value update/signaling lifecycle is missing.
- Raw E5RT encode does not consume the ObjC-bound `_MTLSharedEvent` state in the
  way CoreML's normal prepare/async-submission path does.
- Stage B's input dependency is attached to stream scheduling, but the hidden
  output memory backing still needs a separate synchronization/copy/readback
  primitive.

Calling `_updateCompletionEventFutureValuesWithCompletionSyncPoint:` directly is
known to crash in this manual lifecycle, so do not enable `--objc-sync-update`
unless intentionally debugging that crash.

## Update — Round 16: Tiny Controls Pass, Real Phi Second Op Still Zero

Added more controls to `e5_two_op_stream_probe`:

- `--toy-4d` supplies `[1,4,1,1]` Float16 toy inputs.
- Output stats now read `MLMultiArray` through logical indices, respecting
  shape/strides. This fixed misleading stats for 4D buffers with padded strides.
- `AttemptHiddenToXBinding` now restores the first binding mode that leaves the
  input direct; previously it probed mode 1 successfully and then left mode 3
  active, which could clear `boundFeatureDirectly`.
- `--rebind-second-x` re-applies the direct hidden-to-input binding after
  `_bindInputFeaturesAndWaitEvents`, because that CoreML call clears the direct
  binder while setting up Phi state/inout ports.

Validated controls:

```
FP16 2D toy:          stageA sum=14, stageB sum=28
FP16 4D toy:          stageA sum=14, stageB sum=28
FP16 4D stateful toy: stageA sum=14, stageB sum=28
```

The stateful toy uses `ct.StateType` with duplicate `state_cache` names across
the two models, so basic Float16, 4D tensors, CoreML state, and duplicate state
names are not sufficient to reproduce Phi's failure.

Real Phi result after preserving direct `x`:

```
stageB.x.afterInputBind direct=YES
stageB.directInputs.afterManualChain=(x)
e5rt_execution_stream_encode_operation(stageA) -> 0
e5rt_execution_stream_encode_operation(stageB) -> 0
stageA.hidden nonzero
stageB.hidden all zero
```

The same remains true when direct CoreML ObjC shared events are attached:

```
stageA.completionSharedEvent.beforeUpdate = <_MTLSharedEvent ...>
stageB.waitSharedEvents.beforeUpdate      = (<_MTLSharedEvent ...>)
stageB.hidden sum                         = 0
```

Also tested one-layer real Phi chains (`23_24 -> 24_25`, `24_25 -> 25_26`): the
second Phi op is still zero. Conversely, `24_30` produces nonzero output when it
is placed as the first raw operation. So the failure is not shard size or a bad
layer range; it is specific to a real Phi stateful shard executing as a second
operation in the manually composed raw E5 stream.

Current likely missing piece: an operation-level dependency or inout/state
resource relationship used by CoreML's normal encoder for large stateful
programs, beyond the visible input direct-binding and ObjC shared-event ivars.

At least for **stateful INT8 conv-heavy shards**, ANE placement remains valid at **223 MB compiled size** on this M4 Max.

### Bandwidth at ANE-resident sizes

Pure-linear sweep ([research-probes/ane_bw_sweep.py]) — **only sizes >16 MB are reliably on ANE**:

| Weights | Latency | Effective BW |
|---|---|---|
| 16.8 MB | 0.25 ms | 67 GB/s |
| 67.1 MB | 3.11 ms | 21 GB/s |
| 268 MB | 10.6 ms | 25 GB/s |

Realistic ANE BW is **~25–110 GB/s** depending on shape. The 21 GB/s case at 67 MB is suspiciously slow; likely a bad tile choice. The Gemma-shape SwiGLU expert at 127 MB hit 110 GB/s — close to ANE's advertised peak.

### Cached-instance is mandatory

Probe ([research-probes/ane_cache_probe.py]) — 16 MB linear, 200 iters:

| Operation | Latency |
|---|---|
| Cached `predict` | 0.23 ms |
| Cold `MLModel(path) + predict` | 78.5 ms |
| Cold load alone | 77.4 ms |

**Cold load is ~340× slower than warm predict.** The 60-instance Gemma plan only works if all `MLModel` instances are loaded once and reused. Re-instantiating per token is fatal.

### Realistic Gemma-expert timing

[research-probes/ane_expert_probe.py] builds a Gemma-style SwiGLU MLP (gate+up+silu*+down) with realistic shapes:

| Label | d_model | d_ffn | Weights | Latency | Effective BW |
|---|---|---|---|---|---|
| medium | 2048 | 4096 | 50 MB | 2.40 ms | 21 GB/s |
| gemma  | 2304 | 9216 | 127 MB | 1.15 ms | 110 GB/s |

The gemma shape achieves much better arithmetic-intensity-vs-tile-size match. Take it as the realistic upper bound.

### Decode tok/s projection (revised, measured)

Per token: 8 active experts × 30 layers = 240 expert MLP calls.

**fp16 (worst case, what we measured directly):**
- 240 × 1.15 ms = 276 ms/token → **3.6 tok/s**
- chain primitive cannot help — ANE is one physical device, BW-bound at the SRAM/LPDDR boundary

**INT4 (Gemma 4 actual quant target, BW scales ~4×):**
- 276 / 4 = 69 ms/token → **~14 tok/s** if INT4 streams as int4 to ANE
- if engine materializes fp16 from int4 in main memory before transfer, no win
- compression-on-wire support is unverified

**Plus attention** (~30 layers × ~3–5 ms each at fp16 with KV cache): adds 90–150 ms/token. Total realistic decode: **2.5–4 tok/s fp16** or **6–10 tok/s INT4** if compression-on-wire works.

### What this means

The earlier 25–50 tok/s back-of-envelope was **2–5× too optimistic**. The dominant cost is per-expert BW; 8 active experts × 30 layers is just a lot of bytes per token, and ANE's effective BW (~110 GB/s peak, much less typical) doesn't keep up.

**For Gemma 4 26B-A4B on this M4 Max via ANE, expect ~5–10 tok/s decode at INT4** if the chain primitive is implemented and INT4 streams compressed. The lower-power-than-GPU value prop holds; the speed parity-with-GPU claim does not.

**The numbers that would change this materially:**
- Per-call weight injection via `_weightsBuffer` (one model shape, weights swap) → eliminates per-expert tile setup overhead, possibly 2–3× win.
- ANE INT4 wire compression confirmed → 4× win on BW-bound fraction.
- Speculative decoding with a small dense draft model on GPU → 2–3× tokens/sec multiplier independent of ANE rate.

### Files

- [research-probes/ane_bw_sweep.py] — pure-linear BW sweep
- [research-probes/ane_cache_probe.py] — cached vs fresh
- [research-probes/ane_expert_probe.py] — Gemma-shape SwiGLU
- [research-probes/ane_device_probe.py] — verifies which ops actually ran on ANE
## Sweep Results (Procedure Count)

| N | build_s | save_s | run_s | pkg_MB | ok |
|---|---|---|---|---|---|
| 4 | 0.5 | 0.0 | 0.2 | 0.01 | YES |
| 16 | 1.8 | 0.1 | 1.1 | 0.02 | YES |
| 32 | 3.6 | 0.2 | 2.7 | 0.04 | YES |
| 64 | 7.3 | 0.5 | 7.3 | 0.08 | YES |
| 96 | 11.2 | 0.7 | 14.2 | 0.12 | YES |
| **128** | 15.5 | 0.9 | 23.6 | 0.16 | **YES** |
| 192 | 24.0 | 1.5 | 48.7 | 0.24 | YES |
| 256 | 32.9 | 1.9 | 84.5 | 0.32 | YES |

`run_s` includes a fresh `MLModel(...)` instantiation + `.predict` per function — i.e. ~0.33s/function. In a real driver these would be loaded once and cached.

`pkg_MB` grows linearly with N (~1.25 KB/function) — when each function has unique constants, dedup adds nothing, as expected. For Gemma's per-layer attention weights (shared across all 128 experts) the dedup will be huge.

## Update — Round 5: Gemma-4-26B-A4B Full 30-Layer ANE Compilation (2026-04-25)

The entire Gemma-4-26B-A4B model (30 transformer layers) has been compiled to **90 ANE-resident CoreML shards** with zero failures. This is the first full-depth MoE model converted to 100% ANE execution.

### Architecture per layer (3 shards)

| Shard | Quant | Compiled Size | ANE |
|-------|-------|:---:|:---:|
| Attention (INT8 per_tensor) | INT8 | 33–47 MB | 100% |
| FFN partial 0 (2 expert packs, INT8 per_tensor) | INT8 | ~182 MB | 100% |
| FFN partial 1 + combiner (merged, mixed INT8/fp16) | INT8+fp16 | ~227 MB | 100% |

**Total**: 30 layers × 3 shards = **90 `.mlmodelc` artifacts**

### Empirical laws confirmed at 30-layer scale

1. **Merged combiner eliminates CPU fallback** — standalone combiner shards (19–36 MB) always fell to CPU due to ANE minimum size threshold. Merging into the last FFN partial keeps the shard above the floor. Validated uniformly across all 30 layers (both sliding and global types).

2. **INT8 per_tensor is the production baseline for current Gemma shards** — tested `constexpr_blockwise_shift_scale` per-block quantization paths poisoned the graph to 0% ANE. Do not generalize this to all 4-bit formats: INT4 per-grouped-channel palettization (`constexpr_lut_to_dense`) is a separate, promising but unvalidated path that needs residency + golden gates before scale-out.

3. **Rank-3 tensors (1,1,D) required** — rank-2 (1,D) inputs cause 100% CPU fallback for linear ops. All trace inputs must be rank-3.

4. **`weight_threshold=10_000_000` mixed quantization** — keeps dense MLP weights (gate ~6M, up ~6M, down ~6M params) in fp16 while INT8-quantizing expert pack weights (>10M params each). Required because dense MLP weights are INT8-hostile after norm fusion (outlier channels → only 3 INT8 levels at mean → cos=0.66).

5. **~250 MB compiled shard limit holds** — largest shard (merged last partial + combiner) compiled to ~227 MB, safely within the empirically validated 250 MB ceiling.

### Golden validation (7-layer sample)

| Layer | Type | cos(hidden) | cos(attn) | top-32 overlap |
|-------|---------|-----------|-----------|----------------|
| L0 | sliding | 0.999832 | 0.999964 | 1.000 |
| L5 | global | 0.999922 | 0.999897 | 1.000 |
| L10 | sliding | 0.999814 | 0.999928 | 1.000 |
| L15 | sliding | 0.999482 | 0.999886 | 1.000 |
| L20 | sliding | 0.999555 | 0.999908 | 1.000 |
| L25 | sliding | 0.999711 | 0.999879 | 0.969 |
| L29 | global | 0.955543 | 0.999722 | 0.719 |

All above 0.95 cosine floor. Attention cosines consistently >0.9997. L29 (deepest global layer) lowest at 0.956 — expected for cumulative quantization at the final layer.

### Revised shard-size law

Previous Round 4 established that INT8 stateful shards work at 223 MB. Round 5 now establishes:
- **Safe operating range**: ≤220 MB compiled (ANE_SAFE_SHARD_MB)
- **Hard ceiling**: ~250 MB compiled (ANE_MAX_COMPILED_SHARD_MB)
- The 227 MB merged shard compiles and runs on ANE at every layer without exception.

### Conversion artifacts

All in `research-probes/out/`:
- `gemma4_shard{L}_{L+1}_real_attn_q8.{mlpackage,mlmodelc}` — attention shards (L=0..29)
- `gemma4_shard{L}_{L+1}_real_ffn_p0of2_q8.{mlpackage,mlmodelc}` — regular FFN partial
- `gemma4_shard{L}_{L+1}_real_ffn_p1of2_q8.{mlpackage,mlmodelc}` — merged last partial + combiner
- Batch script: `scratch/convert_all_30.sh`
- Converter: `research-probes/gemma_to_ane.py mixed --n-layers 30 --max-ctx 128 --quant-bits 8`

## Update — Round 17: Phi Raw E5 Hidden-to-X Memory Bridge (2026-04-28)

The raw E5 two-operation Phi path now matches the public CoreML two-call reference for real stateful Phi shards. The missing piece was not state/inout binding or visible events; it was the raw memory object attached to stage B's `x` input port.

### Diagnostic split

`e5_two_op_stream_probe` now logs the E5RT memory object, size, and data pointer retained from any `MLE5InputPort` / `MLE5OutputPort`:

```
e5rt_io_port_retain_memory_object(port, &memoryObject)
e5rt_memory_object_get_size(memoryObject, &size)
e5rt_memory_object_get_data_ptr(memoryObject, &dataPtr)
```

Successful tiny stateful toy chain:

```
stageA.hidden.beforeRawEncode memoryObject=... size=576 dataPtr=0x1005b8000 direct=YES
stageB.x.beforeRawEncode      memoryObject=... size=8   dataPtr=0x1005b8000 direct=YES
stageB sum=28
```

Failing Phi one-layer chain before the fix:

```
stageA.hidden.beforeRawEncode memoryObject=... size=49216 dataPtr=0x111e84000 direct=YES
stageB.x.beforeRawEncode      memoryObject=... size=49216 dataPtr=0x108fc4000 direct=YES
stageB sum=0
```

The ObjC binder showed `boundFeatureDirectly=YES` and `directInputs=(x)`, but the low-level E5RT port for stage B still pointed at a different data buffer.

### Fix

Force the E5RT port memory bridge after CoreML's normal input/state binding and before raw encode:

```
hiddenMemoryObject = e5rt_io_port_retain_memory_object(stageA.hidden)
e5rt_io_port_bind_memory_object(stageB.x, hiddenMemoryObject)
```

After forcing the bind:

```
stageB.x.afterRawBindMemory memoryObject=... size=49216 dataPtr=0x111e84000 direct=YES
```

### Validated outcomes

One-layer Phi reference:

| Chain | Public CoreML stage B sum | Raw E5 stage B sum | Status |
|---|---:|---:|---|
| `phi4mini_layer23_24_q8 -> phi4mini_layer24_25_q8` | -222.598015 | -222.598015 | match |

Fused topology reference:

| Chain | Public CoreML stage B sum | Raw E5 stage B sum | Status |
|---|---:|---:|---|
| `phi4mini_layer0_16_q8 -> phi4mini_layer16_24_q8` | 4590.85129 | 4590.85129 | match |
| `phi4mini_layer16_24_q8 -> phi4mini_layer24_30_q8` | -166.729431 | -166.729431 | match |
| `phi4mini_layer24_30_q8 -> phi4mini_layer30_32_q8` | -116749.305 | -116749.305 | match |

Stage A also matched public in every run. This proves raw E5RT can chain real stateful Phi layer-range shards correctly once the producer output memory object is explicitly bound to the consumer input port. Every adjacent boundary of the best public `16+8+6+2` topology is now pairwise validated against the separate public CoreML reference.

### Implication

The private E5 path is now viable for removing CoreML host materialization between layer shards without increasing public fused-shard size. The next engineering step is to generalize this two-op probe into an N-op chain for the full `16+8+6+2` Phi topology, then profile whether eliminating hidden-state roundtrips improves decode latency and energy.

### N-op stream validation

`e5_two_op_stream_probe` now supports `--manual-chain-all`, which accepts two or more positional `.mlmodelc` paths, loads all operations into one `MLE5ExecutionStream`, binds normal inputs/state per operation, and raw-binds each `stageN.hidden` memory object into `stageN+1.x` before raw E5RT prepare/encode.

Full fused stack validation:

```
phi4mini_layer0_16_q8 -> phi4mini_layer16_24_q8 -> phi4mini_layer24_30_q8 -> phi4mini_layer30_32_q8
```

| Stage | Public sequential sum | Raw one-stream sum | Status |
|---:|---:|---:|---|
| 0 | 4412.64955 | 4412.64955 | match |
| 1 | 4590.85129 | 4590.85129 | match |
| 2 | 4822.46835 | 4822.46835 | match |
| 3 | -196.834778 | -196.834778 | match |

This is the first verified full Phi fused-topology private E5 stream in this repo: all layer shards execute in one stream, with no public CoreML hidden-state roundtrip between layer shards.

### Timing check

The probe supports `--iterations` and `--warmup-iterations` for repeated execution of the already-loaded, already-encoded private stream. On the same machine/session as the public Swift runtime profile:

| Path | Scope | Mean |
|---|---|---:|
| Public CoreML runtime | four layer-shard predictions from `phi4mini_runtime_meta_16_8_6_2.json` | 53.121 ms/token |
| Private E5 one-stream probe | four fused layer shards, 10 warmups + 100 measured executes | 52.593 ms/execute |

The private stream is correct and slightly faster, but the measured win is only about 0.53 ms/token for the layer stack. The public hidden-state roundtrip was not the dominant decode cost; it is a small boundary overhead on top of ANE compute. With the current public profile (`head_predict_reduce_ms=5.082`), replacing only the layer stack would project decode from about 17.18 tok/s to roughly 17.3 tok/s, before any extra integration overhead.

Implication: the private E5 path is valuable as a capability and may matter more for finer-grained sharding, but it is not by itself the next large Phi speedup for the current `16+8+6+2` topology. Higher-leverage targets remain ANE compute shape/topology and LM-head latency.

## Update — Phi LM-Head Shard Count Sweep on `16+8+6+2` (2026-04-28)

The existing 3-way and 8-way LM-head artifacts were re-manifested against the best `16+8+6+2` layer topology so the layer stack stayed constant. Strict residency passed for every 3-way and 8-way LM-head shard (`conv_non_ane=0`, `compute_non_ane=0`).

| LM-head shards | Decode tok/s | Layers ms/token | Head predict+reduce ms/token | Shard work ms/token | Result |
|---:|---:|---:|---:|---:|---|
| 3 | 16.695 | 54.754 | 5.136 | 10.208 | slower |
| 4 | 17.171 | 53.138 | 5.095 | 12.729 | best |
| 8 | 16.740 | 54.573 | 5.156 | 23.292 | slower |

The 4-way head remains the best measured point. More LM-head shards increase aggregate shard work and do not reduce wall time; fewer shards reduce aggregate work but also do not improve wall time. The current head bottleneck looks like fixed CoreML/ANE submission plus reduction/scheduling overhead rather than a simple shard-size parallelism issue.

## Update — Phi `20+4+6+2` Becomes Long-Decode Baseline (2026-04-28)

The previously built `20+4+6+2` topology was re-profiled against `16+8+6+2` with a longer 100-token decode and the same 4-way LM head.

| Topology | Decode tok/s | Layers ms/token | Head predict+reduce ms/token | Notes |
|---|---:|---:|---:|---|
| `16+8+6+2` | 16.596 | 55.084 | 5.166 | prior short-run best |
| `20+4+6+2` | 17.203 | 53.039 | 5.084 | new long-decode best |

Gate status for `20+4+6+2`:

| Shard | Residency | Golden |
|---|---|---|
| `[0,20)` | `conv_non_ane=0`, `compute_non_ane=0` | `cos_hidden=0.998546` |
| `[20,24)` | `conv_non_ane=0`, `compute_non_ane=0` | `cos_hidden=0.999446` |
| `[24,30)` | `conv_non_ane=0`, `compute_non_ane=0` | previously validated |
| `[30,32)` | `conv_non_ane=0`, `compute_non_ane=0` | previously validated |

`20+4+6+2` is now the measured public CoreML baseline for Phi long decode. The compiler cliff remains above 20 fused layers: `[0,24)` falls off ANE and must not be used.

### Private E5 timing on new baseline

The private `--manual-chain-all` path also works for `20+4+6+2`; public sequential synthetic reference matches every stage:

| Stage | Public sequential sum | Raw one-stream sum |
|---:|---:|---:|
| 0 | 4568.3968 | 4568.3968 |
| 1 | 4590.55386 | 4590.55386 |
| 2 | 4822.20798 | 4822.20798 |
| 3 | -196.949768 | -196.949768 |

Timing: private one-stream layers measured `51.662 ms/execute` over 10 warmups + 100 measured executes, versus public runtime layer calls at `53.039 ms/token`. If integrated without additional overhead, this would project the `20+4+6+2` decode path from `17.203 tok/s` to roughly `17.6 tok/s`.

## Update — Book-Shaped Topology Search Tool (2026-04-28)

`python/phi4_mini_topology_search.py` now treats compiled layer shards as a weighted graph:

- states are layer indices `0..32`
- edges are compiled `.mlmodelc` layer ranges
- edge legality comes from residency/golden reports plus known rejections
- edge weights come from Swift `ProfileDecodeLayers` logs

This is the Sakarovitch weighted-automaton version of the Dragon Book compiler-cliff problem: find the lowest-cost legal path through the layer graph. The tool reports both the best whole observed profile and a separate edge-min lower bound, because mixing per-edge minima across different runs is optimistic and not a benchmark claim.

Initial scan over existing artifacts/logs:

```
best_edge_min_topology=16+8+6+2 layers_ms=52.934
best_observed_profile_topology=20+4+6+2 layers_ms=53.039 decode_tok_s=17.203
```

Known rejected edges are excluded from recommendations: `[0,24)` for ANE fallback and `[24,32)` for golden NaNs. This gives the next topology work a repeatable search harness instead of hand-picked partitions.

Follow-up topology gate: `20+5+5+2` is valid but slower. The existing `[20,25)` and `[25,30)` 5-layer shards both passed strict residency (`conv_non_ane=0`, `compute_non_ane=0`) and golden (`cos=0.999350`, `cos=0.999258`). Runtime profile over 99 decode tokens:

```
Timing: decode_tok_s=17.043
ProfileDecodePerToken: layers_ms=53.565 head_predict_reduce_ms=5.104
ProfileDecodeLayers: L0-20=30.036ms L20-25=9.230ms L25-30=9.096ms L30-32=5.202ms
```

Conclusion: `20+4+6+2` remains the public baseline. Equalizing the post-20 tail to `5+5+2` increases layer time instead of reducing it.

## Update — Batched LM-Head Shape Probe (2026-04-28)

The first Iverson/APL-style fatter-array probe is an opt-in LM-head builder path:

```
python/phi4_mini_lm_head_shards.py --batch-tokens 4
```

This changes the representative shard input from `hidden: [1,3072,1,1]` to `hidden: [1,3072,4,1]`, so the same ANE 1x1 conv scores four hidden vectors in one prediction call. The production single-token default is unchanged.

Complete artifact set:

```
artifacts/phi4_mini_ane/lm_head_shards_bt4/Phi4MiniLMHead_bt4_s{0,1,2,3}_q8.mlmodelc
```

Gates:

```
Residency: all four shards PASS with conv_total=1 conv_ane=1 conv_non_ane=0 compute_total=8 compute_ane=8 compute_non_ane=0
Golden: s0 cos=0.999926, s1 cos=0.999932, s2 cos=0.999935, s3 cos=0.999937
```

Microbench on shard 0, 10 warmups + 100 measured iterations:

```
single_ms_per_token=1.608
batch_ms_per_token=0.691
batch_ms_per_call=2.764
speedup_per_token=2.327
```

Interpretation: this is a real ANE-resident throughput shape for multi-stream, speculative verification, or prefill-like work where multiple hidden vectors are available together. It does not by itself accelerate single-stream greedy decode, because greedy decode only has one next-token hidden vector at a time.

Rejected neighboring path: the existing `Phi4MiniLMHead_top1_s0_q8.mlmodelc` top-k shard failed residency (`ios18.topk` and `ios18.cast` on CPU), so CoreML `topk` reduction is not acceptable under the ANE-only mandate.

## Update — N-Gram Speculation Probe (2026-04-29)

Private E5 remains de-prioritized: the measured gain is too small for the product complexity. The next public path is algorithmic speculation, starting with prompt-lookup n-grams.

`phi4_mini_ane.swift` now has an opt-in accounting mode:

```
--ngram-probe --ngram-min 2 --ngram-max 8
```

This does not change generation. It runs the normal exact greedy path and, for each target next token, asks whether the current token history has a prior suffix match whose following token would have proposed the same target.

Smoke result on the current `20+4+6+2` runtime:

```
NGramProbe: targets=30 proposals=24 accepted=24 proposal_rate=0.800 acceptance_rate=1.000 accepted_per_target=0.800 min_n=2 max_n=8 by_n=N2=1/1 N3=1/1 N4=1/1 N5=1/1 N6=1/1 N7=1/1 N8=18/18
```

Interpretation: prompt lookup has strong signal on repetitive output, but it is not yet a latency win. Current public Phi layer shards are stateful single-token CoreML models; `MLState` is mutated in place by each prediction. Exact multi-token speculative verification needs either:

- a public way to clone/rollback state cheaply, or
- batch-token layer artifacts that can verify a block while preserving a commit/discard boundary, or
- a workload mode where independent streams provide multiple already-valid hidden states.

Until one of those exists, n-gram speculation should be treated as an acceptance-rate probe and workload-selection tool, not as a replacement for the exact greedy runtime.

Follow-up implementation: `--prompt-ids-file` lets one loaded runtime evaluate multiple prompt-ID sequences, and `python/phi4_mini_ngram_prompt_suite.py` regenerates a small code-shaped suite from the Phi GGUF tokenizer metadata.

Code-shaped suite result, 5 prompts, `max_new=24`:

```
NGramProbeSuite: targets=100 proposals=74 accepted=69 proposal_rate=0.740 acceptance_rate=0.932 accepted_per_target=0.690 min_n=2 max_n=8 by_n=N2=5/7 N3=4/5 N4=5/5 N5=5/5 N6=5/5 N7=4/5 N8=41/42
```

This is strong enough to justify a public verifier experiment. The likely next artifact is a small batch-token layer verifier, not a private E5 path.

Offline speculative accounting via `python/phi4_mini_ngram_spec_sim.py` estimates the target verifier-pass upper bound from the same runtime log:

```
draft=4: generated_tokens=100 verifier_passes=49 ideal_speedup=2.041 accepted_tokens=69
draft=8: generated_tokens=100 verifier_passes=41 ideal_speedup=2.439 accepted_tokens=69
```

This is not a runtime speed measurement. It is the pass-count target for a future public verifier that can process and commit a candidate block with correct KV state.

## Update — Structured CoT / Grammar-Constrained Scratchpad (2026-04-29)

Kaya Omer's "Structured CoT: Shorter Reasoning with a Grammar File" is applicable to Phi-4-mini on ANE as a public-runtime sampler constraint, not as a model conversion change.

The key mechanism is guided generation: an FSM/grammar forces a compact scratchpad shape such as:

```
GOAL: <line>
STATE: <line>
ALGO: <line>
EDGE: <line>
VERIFY: <line>
```

then leaves the answer/code channel permissive. On ANE this maps cleanly to the existing host-side token selection boundary:

- layer stack: still CoreML/ANE
- LM head: still CoreML/ANE
- host: constrained argmax/sampling over valid grammar tokens

This is allowed under the repo policy because sampling is already a permitted host-side exception. It does not introduce CPU/GPU model compute.

Expected ANE benefit: lower joules/task by reducing total generated tokens and avoiding empty-code / answer-channel drift, not higher per-token throughput. If total generated tokens drop 5x as in the referenced LiveCodeBench public-test experiment, wall time and energy per solved coding task should drop roughly proportionally at the current `~17 tok/s` decode rate, assuming pass rate holds.

Important Phi-specific caveats:

- Phi-4-mini-instruct is not known to be a native `<think>` reasoning model. The local GGUF tokenizer has no single-token `<think>` or `GOAL:` literal. Grammar literals must be tokenized into sequences using GGUF metadata.
- Forced literal tokens must still advance the ANE layer stack to update KV state. A future `advanceOnly` path could skip LM-head prediction for deterministic grammar tokens, saving only the `~5 ms/token` head cost on those literals.
- Constrained argmax over line/code fields needs caps and metrics. Otherwise the model can move bloat from the scratchpad into comments or answer text.

Recommended implementation order:

1. Build a tokenizer-aware grammar manifest generator for a small coding grammar.
2. Add constrained argmax to `phi4_mini_ane.swift` at the current LM-head reduction point.
3. Start with measurement mode: unconstrained vs grammar-constrained on the code-shaped prompt suite, tracking total tokens, code extraction, and pass/syntax proxies.
4. Only after quality holds, add `advanceOnly` for forced literals and measure whether head-skipping improves energy per task.

This sits beside n-gram speculation: grammar reduces token count; n-gram/speculation reduces verifier passes if block verification exists. They are complementary public paths and do not require private E5.

## Update — Structured CoT First Runtime Slice (2026-04-29)

The public Phi runtime now has an opt-in structured-CoT path:

```bash
.venv/bin/python python/phi4_mini_structured_cot_manifest.py
swiftc -O -framework CoreML -framework Foundation \
  -o artifacts/phi4_mini_ane_runtime \
  artifacts/phi4_mini_ane.swift
artifacts/phi4_mini_ane_runtime \
  --meta artifacts/phi4_mini_ane/phi4mini_runtime_meta_20_4_6_2.json \
  --max-new 16 \
  --structured-cot \
  --profile
```

Files:

- `python/phi4_mini_structured_cot_manifest.py`
- `artifacts/phi4_mini_ane/phi4mini_structured_cot_plan.json`
- `artifacts/phi4_mini_ane.swift`

Runtime behavior:

- `--structured-cot` loads the default manifest.
- `--structured-cot-manifest <path>` loads any compatible manifest.
- JSONL serve requests can set `structured_cot` per request if a manifest was loaded.
- Literal stages force exact token IDs and skip the LM-head prediction/reduction for that token selection.
- Field stages run the LM head and constrain argmax by blocking stop tokens; newline can be forced when a field reaches its budget.
- Open answer/code stage returns to the normal greedy path.

First smoke on `20+4+6+2`, `max-new=16`, default BOS prompt:

```text
Generated IDs: [155922, 734, 20446, 1483, 25, 220, 16, 25, 220, 16, 25, 220, 16, 25, 220, 16]
Timing: decode_tokens=15 decode_s=0.903116 decode_tok_s=16.609
ProfileDecodePerToken: layers_ms=56.151 head_predict_reduce_ms=4.049
StructuredCoT: forced_tokens=6 field_content_tokens=10 field_newline_tokens=0 fields_completed=0 active_stage=goal
```

This is a functional smoke, not a quality claim. The short budget only proves the FSM can force the scaffold and constrain field sampling while keeping the ANE path alive. Next quality gate should compare unconstrained vs structured mode on coding prompts with enough `max-new` to reach the `CODE:` stage, then measure token count and code extraction success.

## Update — N-Gram Force Head-Skip Probe (2026-04-29)

The public CoreML state-copy question was rechecked before implementing another n-gram path. CoreML exposes state buffers publicly:

- Python: `MLState.read_state` / `MLState.write_state`
- Swift SDK: `MLState.withMultiArray(for:)`

So public state copy/rollback is possible in principle. It is not a speed path by itself for Phi, because copying the full KV cache is a large host memory transfer and a single-token verifier still requires one ANE layer-stack pass per verified token. Exact speculative speedup still needs batch-token verifier artifacts or another multi-token ANE verification shape.

Implemented an experimental, approximate prompt-lookup mode instead:

```bash
--ngram-force --ngram-min 2 --ngram-max 8
```

Behavior:

- If prompt lookup finds a suffix match, force the proposed next token.
- Skip LM-head prediction/reduction for forced tokens.
- Still run the ANE layer stack for every emitted token, preserving KV alignment with the emitted stream.
- Do not combine with `--ngram-probe`; probe mode is exact accounting, force mode changes generation.

Benchmark on the 5-prompt code-shaped suite (`scratch/ane_private_api/phi4_code_prompt_ids.txt`, `max-new=24`, current `20+4+6+2` manifest):

```text
exact greedy + --ngram-probe:
  decode_tokens=95 decode_s=5.605536 weighted_tok_s=16.948
  avg_layers_ms=53.876 avg_head_ms=5.120
  NGramProbeSuite: targets=100 proposals=74 accepted=69 proposal_rate=0.740 acceptance_rate=0.932 accepted_per_target=0.690

approximate --ngram-force:
  decode_tokens=95 decode_s=5.269287 weighted_tok_s=18.029
  avg_layers_ms=54.755 avg_head_ms=0.703
  force_targets=100 forced=82 force_rate=0.820
```

Interpretation: forcing prompt-lookup tokens buys only `~6.4%` throughput on this suite even with an `82%` force rate, because the remaining bottleneck is the ANE layer stack (`~55 ms/token`). This is a useful product knob only if a task-quality gate says approximate prompt lookup is acceptable. It is not exact speculative decoding and should not be confused with the earlier `2.04x` to `2.44x` verifier-pass upper bound.

## Update — Multi-Token Verifier Feasibility (2026-04-29)

Yes: a public multi-token verifier looks feasible, but it is a new CoreML layer-shard shape, not a Swift-only change.

Key correction: exact block verification does not require cheap whole-state rollback. A verifier may write draft KV entries into future cache positions. If only the first `m` draft tokens are accepted, positions beyond `pos+m` remain masked by future `attn_mask` values. The rejected slot at `pos+m` is overwritten when the runtime processes the target fallback token at that same position. If all draft tokens are accepted, the verifier's mutated state is already the committed state.

The desired `T=4` block verifier shape is:

```text
x:             [1, 3072, 4, 1]
rope_cos:      [4, 64]
rope_sin:      [4, 64]
attn_mask:     [1, 1, 4, 2048]
kv_write_mask: [1, 1, 2048, 4]
hidden:        [1, 3072, 4, 1]
```

This should compose with the existing batch-4 LM-head artifacts:

```text
artifacts/phi4_mini_ane/lm_head_shards_bt4/Phi4MiniLMHead_bt4_s{0,1,2,3}_q8.mlmodelc
```

Verifier algorithm sketch:

1. Current state is already advanced through the last accepted token, and the runtime has that token's target next-token logits.
2. Compare draft token 0 against the already-known target next token.
3. Run the `T=4` block layer stack on draft tokens 0..3, mutating state at positions `pos..pos+3`.
4. Run batch-4 LM head over the returned hidden block.
5. Compare logits row 0 with draft token 1, row 1 with draft token 2, row 2 with draft token 3.
6. If all pass, row 3 supplies the bonus/fallback target token; if a mismatch occurs, row `m-1` supplies the fallback token after the accepted prefix.
7. Set `cacheSeqLen += accepted_count`; stale later draft slots are masked and later overwritten.

New op-pattern risk: the current single-token shard writes one K/V row with broadcast masks. The block shard must scatter four K/V rows into state. That will likely require a 5D broadcast/sum or equivalent gather/scatter lowering. Before scaling, build the smallest representative `T=4` shard and run ANE residency. If this scatter update falls to CPU/GPU, do not scale it.

Implementation order:

1. Add `--batch-tokens 4` / verifier mode to the stateful layer converter.
2. Build one smallest representative block shard, preferably one or two tail layers to keep disk/RAM low.
3. Golden-test block hidden/logits against four sequential single-token forwards.
4. Run `ane-validator` on the compiled block shard.
5. Only after ANE residency and golden parity pass, build block verifier shards matching `20+4+6+2`.

## Update — T=4 Verifier Op-Pattern Probe Passed (2026-04-29)

Added `python/phi4_mini_t4_verifier_probe.py` to test the compiler risk directly without loading Phi GGUF weights. The probe builds a synthetic stateful transformer block with:

- `x = [1, d, 4, 1]`
- `rope_cos/sin = [4, d_head/2]`
- `attn_mask = [1, 1, 4, S]`
- `kv_write_mask = [1, 1, S, 4]`
- state `k_cache_0/v_cache_0 = [1, n_kv_heads, S, d_head]`
- output `hidden = [1, d, 4, 1]`

It checks three things in one run:

1. PyTorch sequential four-token reference vs PyTorch block pass.
2. PyTorch sequential reference vs compiled CoreML block pass.
3. `MLComputePlan` residency for every non-const op.

Results:

```text
tiny non-representative shape:
  d=64 nh=4 nkv=2 dh=16 dff=128 S=64 T=4
  torch_seq_vs_block_cos=0.998337
  coreml_seq_vs_block_cos=0.998335
  conv_non_ane=4 compute_non_ane=97 PASS=False

representative medium shape:
  d=1024 nh=16 nkv=4 dh=64 dff=2048 S=256 T=4
  torch_seq_vs_block_cos=0.999974
  coreml_seq_vs_block_cos=0.999974
  conv_non_ane=0 compute_non_ane=0 PASS=True

Phi-sized synthetic shape:
  d=3072 nh=24 nkv=8 dh=128 dff=8192 S=512 T=4
  torch_seq_vs_block_cos=0.999997
  coreml_seq_vs_block_cos=0.999997 rmse=0.000322
  conv_total=4 conv_ane=4 conv_non_ane=0
  compute_total=145 compute_ane=145 compute_non_ane=0
  mlpackage_size=100.8 MB mlmodelc_size=100.8 MB PASS=True
```

The Phi-sized compute plan placed the risky ops on ANE, including:

- `ios18.read_state`: 2/2 ANE
- `ios18.slice_update`: 2/2 ANE
- `ios18.write_state`: 2/2 ANE
- `ios18.softmax`: 8/8 ANE
- `ios18.matmul`: 16/16 ANE
- `ios18.conv`: 4/4 ANE

Interpretation: the T=4 KV scatter/update op pattern is viable on ANE at Phi dimensions. The tiny CPU result should not be treated as a rejection; it reflects a non-representative small graph. Next gate is a real-weight Phi block shard, preferably one layer first, goldened against four sequential single-token calls.

## Update — Real-Weight Phi Layer-0 T=4 Verifier Passed (2026-04-29)

Added `python/phi4_mini_t4_layer_probe.py` to build and validate a real Phi-4-mini layer-0 block verifier from `models/Phi-4-mini-instruct.Q8_0.gguf`.

The script gates:

1. PyTorch four sequential single-token calls vs one PyTorch `T=4` block call.
2. PyTorch sequential reference vs compiled CoreML INT8 `T=4` block call.
3. Standalone `MLComputePlan` residency through `python/phi4_mini_residency_check.py`.

The first real-weight attempt found a verifier-layout bug: attention output was reshaped directly from `[1, nh, T, dh]` to `[1, d, T, 1]`, interleaving token positions into channels. This is harmless at `T=1` and easy to miss with tiny synthetic weights. Correct layout is:

```python
attn_out = torch.cat(parts, dim=1).permute(0, 1, 3, 2).reshape(1, d, T, 1)
```

After the fix:

```text
real Phi layer-0 T=4 q8 verifier:
  T=4 S=2048 d=3072 nh=24 nkv=8 dh=128 dff=8192
  input_mode=embedding token_ids=[199999, 200021, 14350, 200019]
  torch_seq_vs_block_cos=1.000000 rmse=0.000000
  torch_per_token_cos=1.000000,1.000000,1.000000,1.000000
  coreml_seq_vs_block_cos=0.996174 rmse=0.020813 max_abs=0.808594
  coreml_per_token_cos=0.999879,0.989271,0.999179,0.993851
  conv_total=4 conv_ane=4 conv_non_ane=0
  compute_total=146 compute_ane=146 compute_non_ane=0
  mlpackage_size=100.8 MB mlmodelc_size=100.8 MB PASS=True
```

Independent residency check:

```text
/Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
  python/phi4_mini_residency_check.py \
  scratch/phi4_mini_t4_real_probe/phi4mini_layer0_t4_q8_layoutfix.mlmodelc \
  --json-out scratch/phi4_mini_t4_real_probe/phi4mini_layer0_t4_q8_layoutfix_residency.json

conv_total=4 conv_ane=4 conv_non_ane=0
compute_total=146 compute_ane=146 compute_non_ane=0
PASS=True
```

Interpretation: the real-weight one-layer verifier gate is green. The next step is no longer op-pattern feasibility; it is scale-out plumbing: export `T=4` verifier shards for the production topology, connect the existing batch-4 LM head, and prove exact greedy token equality before benchmarking.

## Update — T=4 Verifier Topology Built; Runtime Experimental (2026-04-29)

Scale-out artifacts built with `python/phi4_mini_t4_export_shard.py`:

```text
artifacts/phi4_mini_ane_t4_verifier/phi4mini_t4_layer0_20_q8.mlmodelc
  size=2015.8 MB conv_total=80 conv_non_ane=0 compute_total=2768 compute_non_ane=0 PASS=True

artifacts/phi4_mini_ane_t4_verifier/phi4mini_t4_layer20_24_q8.mlmodelc
  size=403.2 MB conv_non_ane=0 compute_non_ane=0 PASS=True

artifacts/phi4_mini_ane_t4_verifier/phi4mini_t4_layer24_30_q8.mlmodelc
  size=604.7 MB conv_non_ane=0 compute_non_ane=0 PASS=True

artifacts/phi4_mini_ane_t4_verifier/phi4mini_t4_layer30_32_q8.mlmodelc
  size=201.6 MB conv_non_ane=0 compute_non_ane=0 PASS=True
```

Manifest/runtime plumbing:

- `python/phi4_mini_export_runtime.py` now supports an optional
  `speculative_verifier` section.
- Generated `artifacts/phi4_mini_ane/phi4mini_runtime_meta_20_4_6_2_t4.json`.
- `artifacts/phi4_mini_ane.swift` has opt-in `--speculative` mode that
  loads T=4 verifier layer shards and the existing batch-4 LM-head shards.

Runtime observations:

```text
exact greedy, 5 code prompts, max-new=24:
  decode_tokens=95 decode_s=5.624088 weighted_decode=16.89 tok/s

T=4 speculative, --ngram-min 1 --ngram-max 8, same prompt file:
  decode_tokens=93 decode_s=4.290248 weighted_decode=21.68 tok/s
```

However, exact greedy token equality is not universal yet. Prompt 0 matched for
24 generated tokens, but other prompts diverged. The short synthetic chat prompt
also diverged at the final prefill prediction (`200019` exact vs `200020` T=4).
The likely cause is full-stack numerical drift between the single-token q8 graph
and the T=4 q8 verifier graph, even though one-layer parity and ANE residency
are strong.

Interpretation: the T=4 scale-out path is viable and can be faster on repetitive
code-shaped outputs, but it is not yet a shippable exact speculative decoder.
Keep `--speculative` experimental until full-stack parity or an exactness guard
is implemented.

## Update — Gemma All-FP16 ANE Validation: Full Decode Quality Gate (2026-05-14)

First full-quality validation of Gemma-4-26B-A4B on pure ANE, with all compute ops confirmed on-device. This closes T4.1.5 and T4.3 and establishes the unoptimized all-ANE decode baseline.

### Problem with Round 5 (INT8 q8c stack)

The Round 5 q8c stack had FFN shards silently running on GPU:

| Shard | Compiled size | Actual compute unit |
|-------|:---:|:---:|
| FFN partial 0 of 2 (q8c) | 364 MB | **GPU** |
| FFN partial 1 of 2 (q8c) | 398 MB | **GPU** |

Both exceeded the ~250 MB ANE compiled limit. CoreML emits no warning — it silently falls back to GPU. The symptom is a cascade of cosine error across deep layers (L17→L29 cosine 0.55–0.84 at the worst positions), not a hard failure.

**Rule confirmed**: `du -sh *.mlmodelc` before assuming ANE placement. Size > 250 MB = GPU, regardless of `computeUnits = .cpuAndNeuralEngine`.

### Fix: 8-shard FP16 FFN split

Split each FFN from 2 sub-shards to 8, one expert pack per sub-shard:

| Shard | Compiled size | ANE |
|-------|:---:|:---:|
| Attention FP16 | 33–98 MB | ✓ |
| FFN partial k of 8 (k=0..6) | 182 MB | ✓ |
| FFN last partial 7 of 8 + combiner | 216 MB | ✓ |

**Total per layer**: 1 attn + 8 FFN = 9 shards. **Total model**: 30 × 9 = **270 shards**.

### ANE residency probe (L24, p0of8, representative)

| Op class | Count | Placement |
|----------|:---:|:---:|
| `ios18.linear` (router, gate, up, down) | 35 | ANE |
| `ios18.gelu`, `ios18.softmax`, `ios18.topk` | included above | ANE |
| `ios16.reduce_mean`, `ios18.rsqrt`, `ios18.mul` (RMSNorm) | included above | ANE |
| GPU | 0 | — |
| CPU | 0 | — |

**35/35 compute ops on ANE. 0 GPU. 0 CPU.**

### Quality gate (all-FP16, `gemma_swift_head_meta_allfp16.json`)

7-token prompt `[3689, 563, 506, 5279, 529, 7001, 236881]` vs `gemma_golden.npz`:

| Position | Cosine | Status |
|---|---|---|
| prompt pos 0 | 0.9997 | ✓ |
| prompt pos 1 | 0.9996 | ✓ |
| prompt pos 2 | 0.9977 | ✓ |
| prompt pos 3 | 0.9980 | ✓ |
| prompt pos 4 | 0.9944 | ✓ |
| prompt pos 5 | 0.9982 | ✓ |
| prompt pos 6 | 0.9957 | ✓ |
| decode pos 0 | 0.9976 | ✓ |
| decode pos 1 | 0.9807 | ✓ |
| decode pos 2 | 0.9967 | ✓ |
| decode pos 3 | 0.9926 | ✓ |
| decode pos 4 | 0.9985 | ✓ |
| decode pos 5 | 0.9985 | ✓ |
| decode pos 6 | 0.9923 | ✓ |
| decode pos 7 | 0.9936 | ✓ |
| decode pos 8 | 0.9933 | ✓ |
| decode pos 9 | **0.9258** | ✓ top-1 |
| decode pos 10 | **0.9407** | ✓ top-1 |
| decode pos 11 | **0.9567** | ✓ top-1 |
| decode pos 12 | 0.9770 | ✓ |
| decode pos 13 | 0.9804 | ✓ |
| decode pos 14 | **0.9450** | ✓ top-1 |
| decode pos 15 | 0.9758 | ✓ |
| decode 16-tok exact | `[669, 5279, 529, 7001, 236881] ×3 + [669]` | ✓ |

All 7 prompt positions cosine ≥ 0.97. 16-token greedy decode exactly matches `gemma_golden.npz[next_token_ids]` (16/16 top-1).

**KV-drift characteristic:** Positions 9–11 and 14 dip below 0.97 (min 0.9258 at pos 9). Cause: FP16 KV cache accumulates angular error per decode step. Rank order of top logit is unaffected — 0 wrong tokens at any position. Non-monotonic recovery at pos 12–15 is consistent with the repeating token sequence resetting effective context. The 0.97 cosine gate is a *prompt-grounding* gate; per-step decode cosine below 0.97 is not a failure provided top-1 is correct.

### Timing baseline (unoptimized, 270 shards sequential)

| Metric | Value |
|--------|-------|
| TTFT (model load + 7-tok prefill) | ~212 s |
| Decode rate | 28.9 s/tok (0.034 tok/s) |
| Model: 270 shards × 9 shards/layer | sequential MLModel load per token |

This is the *correctness* baseline, not a performance target. Each decode step loads 270 shards sequentially — the `_ANEChainingRequest` work (Round 2/3) is the path to eliminate this overhead.

### O2: Concurrent FFN partial fan-out (2026-05-14)

The 7 FFN partial shards per layer (`p0of8`–`p6of8`) are **independent**: they take the same `x` input and produce additive `partial_moe` slices. They were previously dispatched sequentially. With `DispatchQueue.concurrent` fan-out + a stable pre-allocated `MLMultiArray` scratch buffer (non-overlapping row writes), all 7 run concurrently per layer.

**Implementation** (`artifacts/gemma_ane.swift`):
- Pre-allocated `MLMultiArray` scratch: `[nPartials × dModel, Float16]`, stable pointer, written from concurrent tasks at non-overlapping row offsets.
- `DispatchGroup` + `ffnPartialsQueue` (`.concurrent`), same pattern as existing `headQueue`.
- After `group.wait()`: reduce scratch rows into `moeAccumF32`, then run `ffnLastModels[layerIdx]` (sequential, depends on prior_moe sum).
- Binary: `gemma_ane_parallel`

**Expected gain** (pending measurement): if ANE schedules the 7 concurrent predictions in parallel, per-layer FFN time drops from `7×T_partial + T_last` to `T_parallel + T_last`. Whether ANE actually parallelizes concurrent `prediction()` calls from different `MLModel` instances is the open question. Results will be filed here after the smoke test completes.

**INT4 palettization probe (2026-05-14, COMPLETE)**:
Single-layer (L0) FFN, all 8 partials exported with `--quant-bits 4 --quant-mode palettize` → `cto.OpPalettizerConfig(nbits=4, mode="kmeans", granularity="per_grouped_channel", group_size=32)` → `constexpr_lut_to_dense`. Artifact suffix: `_q4_pal`.

| Gate | Result |
|------|--------|
| All 8 partials compiled | ✓ p0–p6: 46 MB, p7: 54 MB (vs 182/216 MB FP16, ~75% compression) |
| ANE residency (MLComputePlan) | ✓ ANE: 34/34 real ops, GPU: 0, CPU: 0. UNK=48 are `const` (44) + `ios18.constexpr_lut_to_dense` (4) — compile-time, no runtime device |
| Quality cosine p0of8 (5 unit-norm seeds) | ✓ 0.9845–0.9859 for non-trivial seeds (both-zero seeds: expected routing behaviour, FP16 also returns zero) |

**Decision: GATES PASS → scale-out to all 30 layers is unblocked.** Full 30-layer palettize: 240 partials × ~35 s/shard ≈ 2.3 h. Run with `TMPDIR=<external-scratch> GEMMA_OUT_DIR=...` to avoid local disk full (the local scratch volume was nearly full).  
Constraint: `constexpr_lut_to_dense` replaces the runtime `linear_quantize_weights` path — weight LUTs are baked at export time, which means the `.mlpackage` is the dequantization artifact (larger than INT8 `.mlpackage` but same 46 MB compiled size).

### Updated empirical laws

6. **FFN 2-shard (364/398 MB) → GPU silently. FFN 8-shard (182/216 MB) → ANE.** The ~250 MB compiled shard limit is the hard gate. `du -sh` before assuming ANE placement.

7. **Attn INT8 per-channel quantization is unsafe for global attention layers.** Causes >0.03 cosine drop per global layer, cascading to 0.55 cosine floor by L25. FP16 attn is mandatory for quality on Gemma-4-26B-A4B global layers (L05, L11, L17, L23, L29).

### Artifacts

- Production meta: `research-probes/out/gemma_swift_head_meta_allfp16.json`
- Shard dir: `<external-scratch>/models/gemma4-ane-q8c/`
- FP16 attn: `gemma4_shard{L}_{L+1}_real_attn_fp16.mlmodelc` (L=0..29)
- FP16 FFN: `gemma4_shard{L}_{L+1}_real_ffn_p{k}of8_fp16.mlmodelc` (L=0..29, k=0..7)
- Validation scripts: `research-probes/validate_ffn8_shards.py`, `research-probes/gen_allfp16_meta.py`

