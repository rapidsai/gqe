#pragma once

#include "memory_pool.hpp"
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <iostream>
#include <rmm/device_uvector.hpp>
#include <sstream>
#include <stdint.h>

namespace libperfect {

constexpr uint32_t colors[] = {
  0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
constexpr int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                          \
  do {                                                                 \
    int color_id                      = cid;                           \
    color_id                          = color_id % num_colors;         \
    nvtxEventAttributes_t eventAttrib = {0};                           \
    eventAttrib.version               = NVTX_VERSION;                  \
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType             = NVTX_COLOR_ARGB;               \
    eventAttrib.color                 = colors[color_id];              \
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;       \
    eventAttrib.message.ascii         = name;                          \
    nvtxRangePushEx(&eventAttrib);                                     \
  } while (0)
#define POP_RANGE() nvtxRangePop()

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    std::cerr << file << ":" << line << ": GPUassert: " << cudaGetErrorString(code) << "\n";
    if (abort) { exit(code); }
  }
}

#define gpu_assert(ans)                   \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuThrow(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    std::stringstream what;
    what << file << ":" << line << ": GPUassert: " << cudaGetErrorString(code);
    std::cout << what.str() << std::endl;
    throw std::logic_error(what.str());
  }
}

#define gpu_throw(ans)                   \
  do {                                   \
    gpuThrow((ans), __FILE__, __LINE__); \
  } while (0)

inline std::string annotateLine(std::string const& what, const char* file, int line)
{
  std::stringstream new_what;
  new_what << file << ":" << line << ": " << what;
  return new_what.str();
}

#define annotate_line(what) annotateLine((what), __FILE__, __LINE__)

inline auto sizeOfId(cudf::type_id id, const char* file, int line)
{
  try {
    return cudf::size_of(cudf::data_type(id));
  } catch (std::exception const& e) {
    std::stringstream new_what;
    new_what << "Can't get element_size of type " << static_cast<int32_t>(id) << "\n";
    new_what << e.what();
    throw std::invalid_argument(annotateLine(new_what.str(), file, line));
  }
}

#define size_of_id(id) sizeOfId((id), __FILE__, __LINE__)

// Like a regular pointer but it knows if the data is on gpu or pinned
// and it knows this at compile time.
template <typename T, bool ACCESS_FROM_HOST, bool ACCESS_FROM_DEVICE>
class CudaPointer {
 public:
  using pointer         = CudaPointer;
  using element_type    = T;
  using difference_type = std::ptrdiff_t;
  template <typename U>
  using rebind  = CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>;
  CudaPointer() = default;
  __host__ __device__ explicit CudaPointer(T* buffer) : buffer(buffer) {}
  template <class U,
            std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>,
                             bool> = true>  // implicit conversion allowed if the new
                                            // type is the const of the input.
  __host__ __device__ operator CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>() const
  {
    return {buffer};
  }
  template <class U>
  __host__ __device__ explicit operator CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>() const
  {
    return CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>(
      typename std::pointer_traits<T*>::template rebind<U>(buffer));
  }
  // This is a shortcut to use instead of rebind.
  template <class U>
  __host__ __device__ CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE> as() const
  {
    return static_cast<CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>>(*this);
  }
  // This is a shortcut for the static cast to the underlying pointer.
  // It removes the checks against reading gpu memory from the host
  // and vice-versa so we require this to be explicit.
  __host__ __device__ constexpr T* data() { return static_cast<T*>(*this); }
  __host__ __device__ constexpr T const* data() const { return static_cast<T*>(*this); }
  __host__ __device__ T* operator->() const
  {
#if defined(__CUDA_ARCH__)
    static_assert(ACCESS_FROM_DEVICE);
#else
    static_assert(ACCESS_FROM_HOST);
#endif
    return buffer;
  }
  __host__ __device__ T& operator[](difference_type index) const
  {
#if defined(__CUDA_ARCH__)
    static_assert(ACCESS_FROM_DEVICE);
#else
    static_assert(ACCESS_FROM_HOST);
#endif
    return buffer[index];
  }
  __host__ __device__ constexpr auto operator+(difference_type index) const
  {
    return CudaPointer(buffer + index);
  }
  __host__ __device__ constexpr auto operator-(difference_type index) const
  {
    return CudaPointer(buffer - index);
  }
  __host__ __device__ constexpr auto& operator+=(difference_type index)
  {
    buffer += index;
    return *this;
  }
  __host__ __device__ constexpr auto operator-(CudaPointer const& other) const
  {
    return buffer - other.buffer;
  }
  __host__ __device__ constexpr T& operator*() const
  {
    return (*this)[0];  // The check for reading memory is done in the operator.
  }
  __host__ __device__ constexpr CudaPointer& operator++()
  {
    ++buffer;
    return *this;
  }
  // We make this function explicit because it defeats the safety
  // checks that this class does.  We want that to be intentional, as
  // necessary.
  __host__ __device__ explicit operator T*() const { return buffer; }
  __host__ __device__ constexpr CudaPointer operator++(int)
  {
    auto temp = *this;
    ++(*this);
    return temp;
  }
  __host__ __device__ constexpr bool operator<(const CudaPointer& other) const
  {
    return buffer < other.buffer;
  }
  __host__ __device__ constexpr bool operator!=(const CudaPointer& other) const
  {
    return buffer != other.buffer;
  }
  template <typename other_type,
            bool OTHER_ACCESS_FROM_HOST,
            bool OTHER_ACCESS_FROM_DEVICE,
            std::enable_if_t<
              std::is_same_v<std::remove_const_t<other_type>, std::remove_const_t<element_type>>,
              bool> = true>
  void copy_from(
    CudaPointer<other_type, OTHER_ACCESS_FROM_HOST, OTHER_ACCESS_FROM_DEVICE> const& begin,
    CudaPointer<other_type, OTHER_ACCESS_FROM_HOST, OTHER_ACCESS_FROM_DEVICE> const& end,
    cudaStream_t stream)
  {
    copy_from(
      begin.data(), end.data(), get_memcpy_kind(begin.is_host_memory(), is_host_memory()), stream);
  }
  bool is_host_memory() const { return ACCESS_FROM_HOST; }

 private:
  cudaMemcpyKind get_memcpy_kind(bool source_is_host_memory, bool dest_is_host_memory)
  {
    if (source_is_host_memory) {
      if (dest_is_host_memory) {
        return cudaMemcpyHostToHost;
      } else {
        return cudaMemcpyHostToDevice;
      }
    } else {
      if (dest_is_host_memory) {
        return cudaMemcpyDeviceToHost;
      } else {
        return cudaMemcpyDeviceToDevice;
      }
    }
  }
  void copy_from(element_type const* begin,
                 element_type const* end,
                 cudaMemcpyKind kind,
                 cudaStream_t stream)
  {
    gpu_throw(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(data()),
                              begin,
                              (end - begin) * sizeof(element_type),
                              kind,
                              stream));
  }
  T* buffer;
};

// Owns an array of elements of type T in host pinned memory.
template <typename T>
class CudaPinnedArray {
 public:
  constexpr static bool ACCESS_FROM_HOST   = true;
  constexpr static bool ACCESS_FROM_DEVICE = true;
  using pointer                            = CudaPointer<T, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>;
  using const_pointer = CudaPointer<T const, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>;
  using element_type  = T;
  explicit CudaPinnedArray(size_t count)
  {
    buffer = (T*)GlobalMemoryPool::get().allocate(sizeof(T) * count);
  }
  explicit CudaPinnedArray(std::vector<T> const& source_elements)
    : CudaPinnedArray(source_elements.size())
  {
    memcpy(buffer, source_elements.data(), source_elements.size() * sizeof(T));
  }
  // Five rule of five functions below.
  ~CudaPinnedArray() { GlobalMemoryPool::get().deallocate(buffer); }
  CudaPinnedArray(const CudaPinnedArray& other)      = delete;
  CudaPinnedArray& operator=(const CudaPinnedArray&) = delete;
  CudaPinnedArray(CudaPinnedArray&& other) noexcept : buffer(std::exchange(other.buffer, nullptr))
  {
  }
  CudaPinnedArray& operator=(CudaPinnedArray&& other) noexcept
  {
    CudaPinnedArray temp(std::move(other));
    std::swap(buffer, temp.buffer);
    return *this;
  }
  // Returns a pointer to the first element of the memory.
  constexpr pointer get() const noexcept { return pointer(buffer); }
  // Returns a pointer to the first element of the memory.
  constexpr const_pointer cget() const noexcept { return const_pointer(buffer); }
  // Returns a reference to an element in the memory.
  constexpr T& operator[](size_t i) const { return get()[i]; }

 private:
  T* buffer;
};

// Like a CudaPointer but the element type is erased.
template <bool ACCESS_FROM_HOST, bool ACCESS_FROM_DEVICE, typename T>
class CudaBufferPointer {
 public:
  using pointer         = CudaBufferPointer;
  using difference_type = std::ptrdiff_t;
  CudaBufferPointer(T* elements, const cudf::type_id id) : elements(elements), id(id) {}
  template <typename U>
  explicit operator U const*() const
  {
    // assert id == U?
    return static_cast<U const*>(elements);
  }
  template <typename U>
  explicit operator U*()
  {
    // assert id == U?
    return static_cast<U*>(elements);
  }
  template <typename U>
  CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE> as() const
  {
    return CudaPointer<U, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>(static_cast<U*>(elements));
  }
  template <typename U>
  __device__ __host__ U& at(size_t index)
  {
    // assert sizeof(T) == width?
#if defined(__CUDA_ARCH__)
    static_assert(ACCESS_FROM_DEVICE);
#else
    static_assert(ACCESS_FROM_HOST);
#endif
    return static_cast<U*>(elements)[index];
  }
  template <typename U>
  __device__ __host__ const U& at(size_t index) const
  {
    // assert sizeof(T) == width?
#if defined(__CUDA_ARCH__)
    static_assert(ACCESS_FROM_DEVICE);
#else
    static_assert(ACCESS_FROM_HOST);
#endif
    return static_cast<U const*>(elements)[index];
  }
  __device__ __host__ int64_t operator[](size_t index) const
  {
    // Return the raw bits of the element, paying attention to the size of the element
    // but ignoring the type of the element.
    int64_t ret = 0;
    switch (id) {
      case cudf::type_id::INT8: ret = this->at<int8_t>(index); break;
      case cudf::type_id::INT16: ret = this->at<int16_t>(index); break;
      case cudf::type_id::INT32: ret = this->at<int32_t>(index); break;
      case cudf::type_id::INT64: ret = this->at<int64_t>(index); break;
      case cudf::type_id::UINT8: ret = this->at<uint8_t>(index); break;
      case cudf::type_id::UINT16: ret = this->at<uint16_t>(index); break;
      case cudf::type_id::UINT32: ret = this->at<uint32_t>(index); break;
      case cudf::type_id::UINT64: ret = this->at<uint64_t>(index); break;
      case cudf::type_id::FLOAT32: ret = this->at<int32_t>(index); break;
      case cudf::type_id::FLOAT64: ret = this->at<int64_t>(index); break;
      case cudf::type_id::TIMESTAMP_DAYS:
        ret = this->at<int32_t>(index);
        break;  // int32_t days since epoch.
      default:
#if defined(__CUDA_ARCH__)
        // printf("Unknown cudf type in operator[]: %d\n", static_cast<int32_t>(id));
        // ret = 0;
        assert(0);
#else
        std::stringstream what;
        what << "Unknown cudf type in operator[]: " << static_cast<int32_t>(id);
        throw std::invalid_argument(annotate_line(what.str()));
#endif
    }
    return ret;
  }
  void set(size_t index, int64_t value)
  {
    // Set the raw bits of the element, paying attention to the size of the element
    // but ignoring the type of the element.
    switch (id) {
      case cudf::type_id::INT8: this->at<int8_t>(index) = value; break;
      case cudf::type_id::INT16: this->at<int16_t>(index) = value; break;
      case cudf::type_id::INT32: this->at<int32_t>(index) = value; break;
      case cudf::type_id::INT64: this->at<int64_t>(index) = value; break;
      case cudf::type_id::UINT8: this->at<uint8_t>(index) = value; break;
      case cudf::type_id::UINT16: this->at<uint16_t>(index) = value; break;
      case cudf::type_id::UINT32: this->at<uint32_t>(index) = value; break;
      case cudf::type_id::UINT64: this->at<uint64_t>(index) = value; break;
      case cudf::type_id::FLOAT32: this->at<int32_t>(index) = value; break;
      case cudf::type_id::FLOAT64: this->at<int64_t>(index) = value; break;
      case cudf::type_id::TIMESTAMP_DAYS:
        this->at<int32_t>(index) = value;
        break;  // int32_t days since epoch.
      default:
        std::stringstream what;
        what << "Unknown cudf type in set: " << static_cast<int32_t>(id);
        throw std::invalid_argument(annotate_line(what.str()));
    }
  }
  auto element_size() const { return size_of_id(id); }
  cudf::type_id get_id() const { return id; }
  __host__ __device__ constexpr auto operator+(difference_type index) const
  {
    return CudaBufferPointer(reinterpret_cast<char*>(elements) + (index * size_of_id(id)), id);
  }

 private:
  T* elements;
  cudf::type_id id;
};

using CudaGpuBufferPointer         = CudaBufferPointer<false, true, void>;
using ConstCudaGpuBufferPointer    = CudaBufferPointer<false, true, void const>;
using CudaPinnedBufferPointer      = CudaBufferPointer<true, true, void>;
using ConstCudaPinnedBufferPointer = CudaBufferPointer<true, true, void const>;

template <typename T>
void fill_memory(T* data, T value, size_t count);

// Owns an array of elements of type T on GPU.
template <typename T>
class CudaGpuArray {
 public:
  constexpr static bool ACCESS_FROM_HOST   = false;
  constexpr static bool ACCESS_FROM_DEVICE = true;
  using pointer                            = CudaPointer<T, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>;
  using const_pointer     = CudaPointer<T const, ACCESS_FROM_HOST, ACCESS_FROM_DEVICE>;
  using element_type      = T;
  explicit CudaGpuArray() = default;
  explicit CudaGpuArray(size_t count) : buffer(count, rmm::cuda_stream_default) {}
  explicit CudaGpuArray(std::vector<T> const& source_elements)
    : CudaGpuArray(source_elements.size())
  {
    cudaMemcpy(buffer.data(),
               source_elements.data(),
               source_elements.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  }
  // Returns a pointer to the first element of the memory.
  constexpr pointer get() noexcept { return pointer(buffer.element_ptr(0)); }
  // Returns a pointer to the first element of the memory.
  constexpr const const_pointer get() const noexcept
  {
    return const_pointer(buffer.element_ptr(0));
  }
  //// Returns a pointer to the first element of the memory.
  // constexpr const_pointer get() const noexcept {
  //   return const_pointer(buffer.element_ptr(0));
  // }
  //  Returns a reference to an element in the memory.
  constexpr T const& operator[](size_t i) const { return get()[i]; }
  template <int value>
  void fill_byte()
  {
    static_assert(value == 0 || value == -1);
    cudaMemset(buffer.data(), value, buffer.size() * sizeof(T));
  }
  template <typename U>
  void fill(U value)
  {
    static_assert(std::is_same_v<T, U>);
    if (value == 0) {
      fill_byte<0>();
    } else if (value == -1) {
      fill_byte<-1>();
    } else {
      fill_memory(buffer.data(), value, buffer.size());
    }
  }

  void resize(size_t new_count) { buffer.resize(new_count, rmm::cuda_stream_default); }
  size_t numel() const { return buffer.size(); }
  rmm::device_uvector<T>& get_buffer() { return buffer; }

 private:
  rmm::device_uvector<T> buffer;
};

// Like a CudaGpuArray but the element type is erased.  The type is
// stored inside class and can be accessed at run time.
class CudaGpuBuffer {
 public:
  constexpr static bool ACCESS_FROM_HOST   = false;
  constexpr static bool ACCESS_FROM_DEVICE = true;
  explicit CudaGpuBuffer(size_t count, cudf::type_id id)
    : elements(count * size_of_id(id), rmm::cuda_stream_default), id(id)
  {
  }
  auto get()
  {
    return CudaBufferPointer<ACCESS_FROM_HOST, ACCESS_FROM_DEVICE, void>(elements.data(), id);
  }
  auto get() const
  {
    return CudaBufferPointer<ACCESS_FROM_HOST, ACCESS_FROM_DEVICE, const void>(elements.data(), id);
  }
  auto element_size() const { return size_of_id(id); }
  size_t numel() const noexcept { return elements.size() / size_of_id(id); }
  void resize(size_t new_count)
  {
    elements.resize(new_count * size_of_id(id), rmm::cuda_stream_default);
  }
  cudf::type_id get_id() const { return id; }
  template <typename T>
  explicit CudaGpuBuffer(CudaGpuArray<T>&& array)
    : CudaGpuBuffer(std::move(array.get_buffer().release()),
                    cudf::type_to_id<std::remove_const_t<T>>())
  {
  }
  rmm::device_buffer& get_buffer() { return elements; }

 private:
  CudaGpuBuffer(rmm::device_buffer&& elements, cudf::type_id id)
    : elements(std::move(elements)), id(id)
  {
  }
  rmm::device_buffer elements;
  cudf::type_id id;
};

class CudaPinnedBuffer {
 public:
  constexpr static bool ACCESS_FROM_HOST   = true;
  constexpr static bool ACCESS_FROM_DEVICE = true;
  explicit CudaPinnedBuffer(size_t count, cudf::type_id id) : count(count), id(id)
  {
    buffer = (void*)GlobalMemoryPool::get().allocate(count * size_of_id(id));
  }
  // Five rule of five functions below.
  ~CudaPinnedBuffer() { GlobalMemoryPool::get().deallocate(buffer); }
  CudaPinnedBuffer(const CudaPinnedBuffer& other)      = delete;
  CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept
    : buffer(std::exchange(other.buffer, nullptr)),
      count(std::exchange(other.count, 0)),
      id(std::exchange(other.id, cudf::type_id::EMPTY))
  {
  }
  CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept
  {
    CudaPinnedBuffer temp(std::move(other));
    std::swap(buffer, temp.buffer);
    std::swap(count, temp.count);
    std::swap(id, temp.id);
    return *this;
  }
  auto get() { return CudaBufferPointer<ACCESS_FROM_HOST, ACCESS_FROM_DEVICE, void>(buffer, id); }
  auto get() const
  {
    return CudaBufferPointer<ACCESS_FROM_HOST, ACCESS_FROM_DEVICE, const void>(buffer, id);
  }
  void resize(size_t new_count)
  {
    if (new_count > count) {
      std::stringstream what;
      what << "Can't resize from " << count << " to " << new_count;
      throw std::invalid_argument(annotate_line(what.str()));
    }
    count = new_count;
  }
  auto element_size() const { return size_of_id(id); }
  size_t numel() const noexcept { return count; }

 private:
  void* buffer;
  size_t count;
  cudf::type_id id;
};

}  // namespace libperfect
