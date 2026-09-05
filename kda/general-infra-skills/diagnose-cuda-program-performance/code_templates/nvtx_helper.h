// -----------------------------------------------------------------------------
// nvtx_helper.h - NVTX v3 annotations for the native (C++ / CUDA) layer.
//
// TEMPLATE FILE. Copy it into your project, then replace exactly two tokens:
//
//     project   ->  your project identifier, lowercase (namespace, domain name)
//     PROJECT   ->  the same identifier, uppercase (macro prefix, build switch)
//
// After replacing, `grep -n 'project\|PROJECT' nvtx_helper.h` must come back
// empty. Requires C++11 or later; include it from .cpp and .cu alike.
//
// Build
// -----
// NVTX v3 is header-only: there is nothing to link, and `-lnvToolsExt` is a
// leftover from v2. The CUDA Toolkit include path provides both NVTX headers, so
// a translation unit that already compiles against CUDA needs no new flags.
//
// Two API paths, because the headers ship independently. The C++ wrapper
// <nvtx3/nvtx3.hpp> is preferred, but some CUDA and PyTorch container images
// carry only the C API <nvtx3/nvToolsExt.h>, and treating those as "no NVTX"
// would compile away annotation that the image fully supports. This header
// detects the C++ wrapper, falls back to the C API, and only then gives up -
// saying so once via #pragma message. Both paths present the same names and
// macros, so call sites never know which one compiled. `PROJECT_NVTX_CPP_API`
// and `PROJECT_NVTX_C_API` are readable at compile time when you need to know.
//
// What an NVTX range measures
// ---------------------------
// A range around a kernel launch measures the *host* dispatch, not the kernel:
//
//     PROJECT_NVTX_RANGE("launch_reduce");
//     reduce_kernel<<<grid, block, 0, stream>>>(...);   // returns immediately
//
//     host:    |-- range: setup + cudaLaunchKernel --|
//     device:                                          |--- reduce_kernel ---|
//
// Never add cudaDeviceSynchronize() inside a range to make the bar line up with
// the kernel. It serializes the program, destroys the overlap under study, and
// yields a profile unlike the application that shipped. Let Nsight Systems
// correlate the range with the GPU work its enclosed launches produced.
//
// Where to put ranges
// -------------------
// Treat roughly 1 microsecond as the floor: below that the annotation competes
// with the work it describes. Annotate semantic boundaries, nesting them as
//
//     stage / phase
//       transaction / function
//         selected subphase
//           CUDA API and kernel rows   <- the profiler already names these
//
// so do not wrap every launch, allocation, or copy: the CUDA rows carry those
// names and durations already, and a range per launch only hides the ownership
// question NVTX exists to answer. Keep messages stable and low-cardinality -
// "update_kv_cache", not "update_kv_cache seq=317" - and pass the varying
// number as a payload instead, so a tool can aggregate across calls. Building
// the message costs your time even when no profiler is attached: NVTX itself is
// then a near-no-op, but your sprintf is not.
//
// Domains
// -------
// This layer annotates into the domain "project.native", while the host layer
// (see nvtx_helper.py) uses "project". Separate domains keep each layer
// independently filterable in Nsight, at the cost of host and native ranges no
// longer nesting into one visual stack. If a single hierarchy matters more than
// filtering, move both layers to the global NVTX domain: use nvtx3::scoped_range
// (no _in<>) here, and torch.cuda.nvtx there.
//
// Example
// -------
//     #include "nvtx_helper.h"
//
//     void initialize(cudaStream_t stream) {
//       project::nvtx_name_current_thread("pipeline-worker");
//       project::nvtx_name_stream(stream, "pipeline");   // name resources once
//     }
//
//     void launch_pipeline(cudaStream_t stream, Buffers& buffers) {
//       PROJECT_NVTX_FUNC_RANGE();            // message = the function name
//
//       {
//         PROJECT_NVTX_RANGE("prepare", PROJECT_NVTX_CATEGORY(dispatch));
//         prepare(buffers);
//       }
//       {
//         PROJECT_NVTX_RANGE("launch_reduce", PROJECT_NVTX_CATEGORY(kernel),
//                            PROJECT_NVTX_PAYLOAD(buffers.element_count));
//         reduce_kernel<<<grid, block, 0, stream>>>(...);
//       }
//     }
//
// For work whose lifetime does not match a lexical scope - submitted on one
// thread and completed on another - use the handle form instead, which is the
// only correct choice there, because push/pop is a per-thread stack:
//
//     auto handle = project::nvtx_range_start("request");
//     // ... hand the handle to whichever thread finishes the request ...
//     project::nvtx_range_end(handle);
//
// The same code works unchanged inside a PyTorch C++/CUDA extension operator;
// nothing here depends on a framework.
// -----------------------------------------------------------------------------

#pragma once

#include <cstdint>

// Detection is two-tier: the C++ wrapper first, then the C API, then nothing.
// PROJECT_NVTX_AUTODETECTED exists so that an explicit -DPROJECT_NVTX_ENABLED=0
// stays quiet, while finding no NVTX header at all - which is an accident - is
// reported.
#if !defined(PROJECT_NVTX_ENABLED)
#  define PROJECT_NVTX_AUTODETECTED 1
#  if defined(__has_include)
#    if __has_include(<nvtx3/nvtx3.hpp>)
#      define PROJECT_NVTX_ENABLED 1
#      define PROJECT_NVTX_CPP_API 1
#    elif __has_include(<nvtx3/nvToolsExt.h>)
#      define PROJECT_NVTX_ENABLED 1
#      define PROJECT_NVTX_C_API 1
#    else
#      define PROJECT_NVTX_ENABLED 0
#    endif
#  else
#    define PROJECT_NVTX_ENABLED 0
#  endif
#else
#  define PROJECT_NVTX_AUTODETECTED 0
#  if PROJECT_NVTX_ENABLED
#    if defined(__has_include)
#      if __has_include(<nvtx3/nvtx3.hpp>)
#        define PROJECT_NVTX_CPP_API 1
#      elif __has_include(<nvtx3/nvToolsExt.h>)
#        define PROJECT_NVTX_C_API 1
#      else
#        error "PROJECT_NVTX_ENABLED=1 but neither <nvtx3/nvtx3.hpp> nor <nvtx3/nvToolsExt.h> is on the include path"
#      endif
#    else
// Detection is impossible here, and silently disabling an explicitly requested
// NVTX is the failure this header exists to avoid. Pick the C API, which every
// NVTX distribution ships, so a wrong guess fails loudly at the #include.
#      define PROJECT_NVTX_C_API 1
#    endif
#  endif
#endif

#if !defined(PROJECT_NVTX_CPP_API)
#  define PROJECT_NVTX_CPP_API 0
#endif
#if !defined(PROJECT_NVTX_C_API)
#  define PROJECT_NVTX_C_API 0
#endif

// PROJECT_NVTX_ENABLED stays the master switch: disabled means no API path.
#if !PROJECT_NVTX_ENABLED
#  undef PROJECT_NVTX_CPP_API
#  undef PROJECT_NVTX_C_API
#  define PROJECT_NVTX_CPP_API 0
#  define PROJECT_NVTX_C_API 0
#endif

#define PROJECT_NVTX_CONCAT_INNER(a, b) a##b
#define PROJECT_NVTX_CONCAT(a, b) PROJECT_NVTX_CONCAT_INNER(a, b)

namespace project {

/// NVTX domain for this project's native layer. One domain per library or
/// component is the intended granularity; see the "Domains" note above.
struct nvtx_domain {
  static constexpr char const* name{"project.native"};
};

/// Starter category taxonomy, subdividing the domain above. A slash builds a
/// hierarchy that tools can group on ("kernel/attention", "memory/pool"). Edit
/// this list to match the pipeline being instrumented; ids only need to be
/// unique and nonzero within the domain, and these tags stay compile-time only,
/// so they cost nothing when NVTX is disabled.
namespace nvtx_category {

struct dispatch {
  static constexpr char const* name{"dispatch"};
  static constexpr uint32_t id{1};
};
struct memory {
  static constexpr char const* name{"memory"};
  static constexpr uint32_t id{2};
};
struct transfer {
  static constexpr char const* name{"transfer"};
  static constexpr uint32_t id{3};
};
struct synchronization {
  static constexpr char const* name{"synchronization"};
  static constexpr uint32_t id{4};
};
struct kernel {
  static constexpr char const* name{"kernel"};
  static constexpr uint32_t id{5};
};

}  // namespace nvtx_category
}  // namespace project

#if PROJECT_NVTX_ENABLED

#if PROJECT_NVTX_CPP_API
#  include <nvtx3/nvtx3.hpp>
#else
#  include <type_traits>

#  include <nvtx3/nvToolsExt.h>
#endif

// Resource naming and CUDA detection are identical on both API paths, so they
// live here rather than in either branch: keeping one copy is what stops the two
// from drifting apart.
//
// cudaStream_t naming lives in the runtime-API header; nvToolsExtCuda.h only
// names the driver API's CUstream. That header itself includes cuda.h, so the
// driver headers must be reachable too - all three ship with the CUDA Toolkit,
// but a host-only translation unit cannot assume any of them.
#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>) && __has_include(<cuda.h>) && \
      __has_include(<nvtx3/nvToolsExtCudaRt.h>)
#    define PROJECT_NVTX_HAS_CUDA_RUNTIME 1
#  endif
#endif
#if !defined(PROJECT_NVTX_HAS_CUDA_RUNTIME)
#  define PROJECT_NVTX_HAS_CUDA_RUNTIME 0
#endif

#if PROJECT_NVTX_HAS_CUDA_RUNTIME
#  include <cuda_runtime.h>
#  include <nvtx3/nvToolsExtCudaRt.h>
#endif

// nvtxNameOsThreadA takes a native thread id, not a pthread_t, so the id has to
// be fetched per platform.
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
// NOMINMAX before windows.h, or its min/max macros break every std::min and
// std::max in the translation units that include this header.
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sys/syscall.h>
#  include <unistd.h>
#elif defined(__APPLE__)
#  include <pthread.h>
#endif

namespace project {

/// Name the calling OS thread, turning "Thread 192814" into a readable row.
///
/// Call it once, from the thread itself, during setup. Naming is global rather
/// than domain-scoped, so pick names that stay unambiguous process-wide.
inline void nvtx_name_current_thread(char const* name) {
#if defined(_WIN32)
  nvtxNameOsThreadA(static_cast<uint32_t>(::GetCurrentThreadId()), name);
#elif defined(__linux__)
  nvtxNameOsThreadA(static_cast<uint32_t>(::syscall(SYS_gettid)), name);
#elif defined(__APPLE__)
  uint64_t thread_id = 0;
  ::pthread_threadid_np(nullptr, &thread_id);
  nvtxNameOsThreadA(static_cast<uint32_t>(thread_id), name);
#else
  // TODO: no portable way to obtain a native thread id on this platform; add a
  // branch here if thread rows need names in your build.
  static_cast<void>(name);
#endif
}

#if PROJECT_NVTX_HAS_CUDA_RUNTIME
/// Name a CUDA stream, so timeline rows read "attention" instead of "Stream 27".
inline void nvtx_name_stream(cudaStream_t stream, char const* name) {
  nvtxNameCudaStreamA(stream, name);
}
#else
/// Stream naming is unavailable without the CUDA runtime headers in this
/// translation unit; kept as a no-op so call sites compile everywhere.
template <typename Stream>
inline void nvtx_name_stream(Stream, char const*) {}
#endif

}  // namespace project

#endif  // PROJECT_NVTX_ENABLED

#if PROJECT_NVTX_CPP_API

namespace project {

/// Handle for a range whose lifetime crosses lexical scopes or threads.
using nvtx_range_handle = nvtx3::range_handle;

/// Resolve a category tag struct into this domain's registered NVTX category.
///
/// Prefer the PROJECT_NVTX_CATEGORY macro at call sites: it keeps the tag name
/// short and stays valid when NVTX is compiled out.
template <typename Category>
inline nvtx3::named_category_in<nvtx_domain> const& nvtx_category_of() {
  return nvtx3::named_category_in<nvtx_domain>::get<Category>();
}

/// Wrap one scalar as an NVTX payload.
///
/// Payloads are how a varying value rides along with a stable message, instead
/// of being formatted into a per-call string the profiler cannot aggregate.
template <typename Value>
inline nvtx3::payload nvtx_payload_of(Value value) {
  return nvtx3::payload{value};
}

/// Open a range that must be closed explicitly, possibly on another thread.
///
/// Accepts the same arguments as PROJECT_NVTX_RANGE: the message first, then
/// optionally a category and a payload in either order. nvtx3 would accept the
/// message anywhere, but the C fallback below requires it first, so writing it
/// first everywhere keeps call sites compiling under both APIs.
template <typename... Args>
inline nvtx_range_handle nvtx_range_start(Args const&... args) {
  return nvtx3::start_range_in<nvtx_domain>(args...);
}

/// Close a range opened by nvtx_range_start.
inline void nvtx_range_end(nvtx_range_handle handle) {
  nvtx3::end_range_in<nvtx_domain>(handle);
}

/// Emit an instantaneous event rather than an interval.
template <typename... Args>
inline void nvtx_mark(Args const&... args) {
  nvtx3::mark_in<nvtx_domain>(args...);
}

}  // namespace project

/// Annotate the enclosing scope. Arguments: the message first, then optionally
/// a category and a payload in either order. The C fallback path requires the
/// message first, so keep that order even though nvtx3 does not need it.
#define PROJECT_NVTX_RANGE(...)                          \
  ::nvtx3::scoped_range_in<::project::nvtx_domain>       \
      PROJECT_NVTX_CONCAT(project_nvtx_range_, __LINE__) \
  {                                                      \
    __VA_ARGS__                                          \
  }

/// Annotate the enclosing function, using its name as the message. Preferred
/// for repeated instrumentation: the name is registered once, not re-processed
/// on every call.
#define PROJECT_NVTX_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(::project::nvtx_domain)

/// Emit an instantaneous event. Same arguments as PROJECT_NVTX_RANGE.
#define PROJECT_NVTX_MARK(...) ::project::nvtx_mark(__VA_ARGS__)

#elif PROJECT_NVTX_C_API

namespace project {

/// Handle for a range whose lifetime crosses lexical scopes or threads.
using nvtx_range_handle = nvtxRangeId_t;

inline nvtxDomainHandle_t nvtx_domain_handle();

/// One category, resolved and registered on first use.
struct nvtx_c_category {
  uint32_t id;
  char const* name;
};

/// One scalar payload with the NVTX type tag that preserves its meaning.
///
/// Only the field matching ``type`` is meaningful. The C++ path keeps a value's
/// exact width (int32, float); this path widens to int64 or double, which a tool
/// displays identically while keeping one conversion instead of six overloads.
/// Collapsing everything into an unsigned field instead would turn -1 into
/// 18446744073709551615 and truncate 3.75 to 3.
struct nvtx_c_payload {
  int32_t type;
  uint64_t unsigned_value;
  int64_t signed_value;
  double real_value;
};

/// One message string registered with the domain, so a repeatedly annotated
/// site pays the string cost once rather than on every call.
class nvtx_registered_message final {
 public:
  explicit nvtx_registered_message(char const* message)
      : handle_(nvtxDomainRegisterStringA(nvtx_domain_handle(), message)) {}

  nvtxStringHandle_t handle() const { return handle_; }

 private:
  nvtxStringHandle_t handle_;
};

/// Resolve a category tag struct into this domain's registered NVTX category.
///
/// Prefer the PROJECT_NVTX_CATEGORY macro at call sites: it keeps the tag name
/// short and stays valid when NVTX is compiled out.
template <typename Category>
inline nvtx_c_category nvtx_category_of() {
  static bool const registered = []() {
    nvtxDomainNameCategoryA(nvtx_domain_handle(), Category::id, Category::name);
    return true;
  }();
  static_cast<void>(registered);
  return {Category::id, Category::name};
}

/// Wrap one scalar as an NVTX payload, keeping signed, unsigned, and real values
/// distinguishable in the trace.
template <typename Value>
inline nvtx_c_payload nvtx_payload_of(Value value) {
  nvtx_c_payload payload{};
  if (std::is_floating_point<Value>::value) {
    payload.type = NVTX_PAYLOAD_TYPE_DOUBLE;
    payload.real_value = static_cast<double>(value);
  } else if (std::is_signed<Value>::value) {
    payload.type = NVTX_PAYLOAD_TYPE_INT64;
    payload.signed_value = static_cast<int64_t>(value);
  } else {
    payload.type = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    payload.unsigned_value = static_cast<uint64_t>(value);
  }
  return payload;
}

/// Create the domain once per process.
inline nvtxDomainHandle_t nvtx_domain_handle() {
  static nvtxDomainHandle_t domain = nvtxDomainCreateA(nvtx_domain::name);
  return domain;
}

inline void nvtx_set_message(nvtxEventAttributes_t& attributes,
                            char const* message) {
  attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = message;
}

inline void nvtx_set_message(nvtxEventAttributes_t& attributes,
                            nvtx_registered_message const& message) {
  attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attributes.message.registered = message.handle();
}

inline void nvtx_set_payload(nvtxEventAttributes_t& attributes,
                            nvtx_c_payload payload) {
  attributes.payloadType = payload.type;
  switch (payload.type) {
    case NVTX_PAYLOAD_TYPE_DOUBLE:
      attributes.payload.dValue = payload.real_value;
      break;
    case NVTX_PAYLOAD_TYPE_INT64:
      attributes.payload.llValue = payload.signed_value;
      break;
    default:
      attributes.payload.ullValue = payload.unsigned_value;
      break;
  }
}

/// Build event attributes from a message, and optionally a category and a
/// payload in either order — the argument freedom the C++ path gets for free.
template <typename Message>
inline nvtxEventAttributes_t nvtx_attributes(Message const& message) {
  nvtxEventAttributes_t attributes{};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  nvtx_set_message(attributes, message);
  return attributes;
}

template <typename Message>
inline nvtxEventAttributes_t nvtx_attributes(Message const& message,
                                            nvtx_c_category category) {
  auto attributes = nvtx_attributes(message);
  attributes.category = category.id;
  return attributes;
}

template <typename Message>
inline nvtxEventAttributes_t nvtx_attributes(Message const& message,
                                            nvtx_c_payload payload) {
  auto attributes = nvtx_attributes(message);
  nvtx_set_payload(attributes, payload);
  return attributes;
}

template <typename Message>
inline nvtxEventAttributes_t nvtx_attributes(Message const& message,
                                            nvtx_c_category category,
                                            nvtx_c_payload payload) {
  auto attributes = nvtx_attributes(message, category);
  nvtx_set_payload(attributes, payload);
  return attributes;
}

template <typename Message>
inline nvtxEventAttributes_t nvtx_attributes(Message const& message,
                                            nvtx_c_payload payload,
                                            nvtx_c_category category) {
  return nvtx_attributes(message, category, payload);
}

/// RAII range, so an early return or a throw cannot leave a range open.
class NvtxRange final {
 public:
  template <typename Message, typename... Args>
  explicit NvtxRange(Message const& message, Args const&... args)
      : domain_(nvtx_domain_handle()) {
    auto attributes = nvtx_attributes(message, args...);
    nvtxDomainRangePushEx(domain_, &attributes);
  }

  ~NvtxRange() { nvtxDomainRangePop(domain_); }

  NvtxRange(NvtxRange const&) = delete;
  NvtxRange& operator=(NvtxRange const&) = delete;

 private:
  nvtxDomainHandle_t domain_;
};

/// Open a range that must be closed explicitly, possibly on another thread.
template <typename Message, typename... Args>
inline nvtx_range_handle nvtx_range_start(Message const& message,
                                          Args const&... args) {
  auto attributes = nvtx_attributes(message, args...);
  return nvtxDomainRangeStartEx(nvtx_domain_handle(), &attributes);
}

/// Close a range opened by nvtx_range_start.
inline void nvtx_range_end(nvtx_range_handle handle) {
  nvtxDomainRangeEnd(nvtx_domain_handle(), handle);
}

/// Emit an instantaneous event rather than an interval.
template <typename Message, typename... Args>
inline void nvtx_mark(Message const& message, Args const&... args) {
  auto attributes = nvtx_attributes(message, args...);
  nvtxDomainMarkEx(nvtx_domain_handle(), &attributes);
}

}  // namespace project

/// Annotate the enclosing scope. Arguments: the message first, then optionally
/// a category and a payload in either order.
#define PROJECT_NVTX_RANGE(...)                            \
  ::project::NvtxRange PROJECT_NVTX_CONCAT(project_nvtx_range_, __LINE__)(__VA_ARGS__)

/// Annotate the enclosing function, using its name as the message. The name is
/// registered once per site, matching what the C++ path's function macro does.
#define PROJECT_NVTX_FUNC_RANGE()                                              \
  static ::project::nvtx_registered_message const project_nvtx_func_message__{ \
      __func__};                                                               \
  ::project::NvtxRange project_nvtx_func_range__ {                             \
    project_nvtx_func_message__                                                \
  }

/// Emit an instantaneous event. Same arguments as PROJECT_NVTX_RANGE.
#define PROJECT_NVTX_MARK(...) ::project::nvtx_mark(__VA_ARGS__)

#else  // no NVTX API selected

#if PROJECT_NVTX_AUTODETECTED
#pragma message( \
    "project: neither <nvtx3/nvtx3.hpp> nor <nvtx3/nvToolsExt.h> was found, so native NVTX annotations compile to nothing and no native ranges will appear in a profile. Add the CUDA Toolkit include path, vendor NVTX release-v3, or define PROJECT_NVTX_ENABLED=0 to silence this.")
#endif

namespace project {

/// Stand-in for a category or payload argument while NVTX is compiled out.
struct nvtx_disabled_argument {};

/// Stand-in handle, so call sites keep compiling with NVTX disabled.
struct nvtx_range_handle {};

template <typename Category>
inline nvtx_disabled_argument nvtx_category_of() {
  return {};
}

template <typename Value>
inline nvtx_disabled_argument nvtx_payload_of(Value) {
  return {};
}

template <typename... Args>
inline nvtx_range_handle nvtx_range_start(Args const&...) {
  return {};
}

inline void nvtx_range_end(nvtx_range_handle) {}

template <typename... Args>
inline void nvtx_mark(Args const&...) {}

inline void nvtx_name_current_thread(char const*) {}

template <typename Stream>
inline void nvtx_name_stream(Stream, char const*) {}

}  // namespace project

#define PROJECT_NVTX_RANGE(...) static_cast<void>(0)
#define PROJECT_NVTX_FUNC_RANGE() static_cast<void>(0)
#define PROJECT_NVTX_MARK(...) static_cast<void>(0)

#endif  // selected NVTX API

/// Name a category by its tag, e.g. PROJECT_NVTX_CATEGORY(dispatch).
#define PROJECT_NVTX_CATEGORY(tag) \
  ::project::nvtx_category_of<::project::nvtx_category::tag>()

/// Attach one scalar to an event, e.g. PROJECT_NVTX_PAYLOAD(byte_count).
#define PROJECT_NVTX_PAYLOAD(value) ::project::nvtx_payload_of(value)
