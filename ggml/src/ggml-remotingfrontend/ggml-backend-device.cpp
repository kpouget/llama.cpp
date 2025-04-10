#include "ggml-remoting.h"

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

static ggml_backend_buffer_type_t ggml_backend_remoting_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_remoting_host_buffer_type();
}


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

const struct ggml_backend_device_i ggml_backend_remoting_device_i = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_remoting_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
