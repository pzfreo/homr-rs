// Benchmark-only Node N-API bridge to macOS's retired-instruction counter.
// cc -bundle -undefined dynamic_lookup -I/opt/homebrew/include/node bench/counter.c -o /scratch/counter.node
#include <node_api.h>
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>
static napi_value instructions(napi_env env, napi_callback_info info) {
    (void)info;
    struct rusage_info_v4 usage;
    if (proc_pid_rusage(getpid(),RUSAGE_INFO_V4,(rusage_info_t *)&usage)) {
        napi_throw_error(env,0,"proc_pid_rusage failed");return 0;
    }
    napi_value result;napi_create_double(env,(double)usage.ri_instructions,&result);return result;
}
static napi_value init(napi_env env,napi_value exports) {
    napi_value fn;napi_create_function(env,"instructions",NAPI_AUTO_LENGTH,instructions,0,&fn);
    napi_set_named_property(env,exports,"instructions",fn);return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME,init)
