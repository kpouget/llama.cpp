#include "ggml-remoting-frontend.h"

#include <ostream>
#include <iostream>
#include <mutex>

#include <chrono>
#include <thread>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#define RMT_LOG_DEBUG(msg) std::cerr << msg << std::endl

#define UNUSED GGML_UNUSED

int ggml_backend_remoting_get_device_count();

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_remoting_get_device_count();
}

static const char * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_REMOTING_NAME;
}

struct ggml_backend_remoting_device_context {
    size_t device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "API Remoting";
}

static const char * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "API Remoting device";
}

static enum ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_remoting_device_get_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    UNUSED(device);
    *total = 1024*1024*1024;
    *free = *total;
}

// vk buffer type
static const char * ggml_backend_remoting_buffer_type_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);

    return "Remoting buffer";
}

static ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
  UNUSED(buft);
  UNUSED(size);

  return nullptr;
}

static size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 4096;
}

static size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 40960;
}

static size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    UNUSED(tensor);
    return ggml_nbytes(tensor);
}

struct ggml_backend_remoting_buffer_type_context {
    std::string name;
};

static ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {

    static struct ggml_backend_buffer_type buft {
      /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
      /* .device   = */ dev,
      /* .context  = */ new ggml_backend_remoting_buffer_type_context{ "device_name"},
    };

    return & buft;
}

static bool ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  UNUSED(dev);
  UNUSED(op);

  return true;
}

static bool ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    UNUSED(dev);
    UNUSED(buft);
    return true;
}


static bool ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    UNUSED(dev);
}

static const char * ggml_backend_remoting_name(ggml_backend_t backend) {
    UNUSED(backend);

    return "API Remoting backend";
}

static void ggml_backend_remoting_free(ggml_backend_t backend) {
    UNUSED(backend);
}

static ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    UNUSED(backend);
    UNUSED(cgraph);

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ NULL,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ NULL,  // ggml_backend_remoting_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ NULL,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static void ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_guid_t ggml_backend_remoting_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    return &guid;
}


static ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *)dev->context;

    ggml_backend_t remoting_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_remoting_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    return remoting_backend;
}


static const struct ggml_backend_device_i ggml_backend_remoting_device_i = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ NULL, //ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL, //ggml_backend_remoting_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (size_t i = 0; i < ggml_backend_remoting_reg_get_device_count(reg); i++) {
                ggml_backend_remoting_device_context * ctx = new ggml_backend_remoting_device_context;
                char desc[256] = "API Remoting device";

                ctx->device = i;
                ctx->name = GGML_REMOTING_NAME + std::to_string(i);
                ctx->description = desc;
                devices.push_back(new ggml_backend_device {
                    /* .iface   = */ ggml_backend_remoting_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    return devices[device];
}

int ggml_backend_remoting_get_device_count() {
    return 1;
}

static const struct ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_remoting_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ nullptr,
    };

    RMT_LOG_DEBUG("ggml_backend_remoting_frontend_reg() hello :wave:");
    return &reg;
}
