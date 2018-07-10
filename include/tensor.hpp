#pragma once

#include <iterator>
#include <clrt.hpp>
#include <size.hpp>
#include <utilities.hpp>

namespace clRT {

class Tensor {
public:
  Tensor(cl::Context context, const Size& shape, const Size& padding, const Size& alignment, Type type = Type::clrt_fp32)
  : _context(context)
  , _shape(shape)
  , _padding(padding)
  , _alignment(alignment)
  , _internal(align(shape + 2 * padding, alignment))
  , _type(type)
  {
    update_buffer_layout();
  }

  Tensor(cl::Context context, const Size& size, const Size& padding, Type type = Type::clrt_fp32)
  : Tensor(context, size, padding, Size::Fill(size.size(), 1), type) {}

  Tensor(cl::Context context, const Size& size, Type type = Type::clrt_fp32)
  : Tensor(context, size, Size::Zeros(size.size()), Size::Fill(size.size(), 1), type) {}

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.to_string();
    return os;
  }

  std::string to_string() const {
    std::stringstream ret;
    ret << "Tensor shape = " << _shape << " padding = " << _padding << " alignment = " << _alignment << " ";
    ret << "Type = " << type_name(_type);
    // ret << (is_allocated() ? "Allocated" : "Not Allocated") << ", ";
    // ret << (is_mapped() ? "Mapped" : "Unmapped");
    return ret.str();
  }

  Size shape() {
    return _shape;
  }

  Size padding() {
    return _padding;
  }

  Size alignment() {
    return _alignment;
  }

  Size pitch() {
    return _pitch;
  }

  Size internal() {
    return _internal;
  }

  size_t buffer_size() {
    return _buffer_size;
  }

  cl_int allocate(cl_mem_flags flags = CL_MEM_READ_WRITE, void* host_ptr = nullptr) {
    if (_allocated)
      throw std::runtime_error("Tensor::allocate");
    cl_int err;
    _buffer = cl::Buffer(_context, flags, _buffer_size, host_ptr, &err);
    _allocated = (err == CL_SUCCESS);
    return err;
  }

  bool allocated() {
    return _allocated;
  }

  void* map(cl::CommandQueue q, bool blocking = false, bool read_only = false, cl_int *err = nullptr) {
    if (!_mapped) {
      //_mapped_ptr = q.enqueueMapBuffer(_buffer, blocking, read_only ? CL_MAP_READ : CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, err);
      _mapped_read_only = read_only;
      _mapped = (err == CL_SUCCESS);
    }
    return _mapped_ptr;
  }

protected:
  void update_buffer_layout() {
    if (_internal.size() > 0) {
      _pitch = Size::Zeros(_internal.size());
      _pitch[0] = type_size(_type);
      _buffer_size = type_size(_type) * _internal[0];
      for (int i = 1; i < _internal.size(); ++i) {
        _pitch[i] = _pitch[i - 1] * _internal[i - 1];
        _buffer_size *= _internal[i];
      }
    } else {
      _buffer_size = 0;
    }
  }

private:
  cl::Context _context;
  Size _shape;
  Size _padding;
  Size _alignment;
  Size _internal;
  Type _type;

  Size _pitch;
  size_t _buffer_size;
  bool _allocated = false;
  cl::Buffer _buffer;
  bool _mapped = false;
  bool _mapped_read_only = false;
  void* _mapped_ptr = nullptr;
};





//   cl_int unmap(cl::CommandQueue q, bool blocking = false) {
//     cl_int err = CL_SUCCESS;
//     if (mapped) {
//       cl_int err = q.enqueueUnmapMemObject(buffer, mapped_ptr);
//       if (blocking) {
//         q.finish();
//       }
//       mapped = false;
//     }
//     return err;
//   }

//   bool is_mapped() const {
//     return mapped;
//   }

//   bool is_mapped_read_only() {
//     return mapped_read_only;
//   }


// private:
//   cl::Context context;
//   Type type;
//   Size size;       // tensor size (elements)
//   Size padding;    // padding around the tensor (elements)
//   Size alignment;  // tensor aligment (elements)
//   Size total_size; // tensor align(size + 2 * padding, alignment) (elements)
//   size_t buffer_size;             // opencl flatten buffer size (bytes)

//   bool allocated = false;
//   bool mapped = false;
//   bool mapped_read_only = false;
//   void* mapped_ptr = nullptr;
//   cl_mem_flags flags = 0;
//   cl::Buffer buffer;
// };

} // namespace clRT