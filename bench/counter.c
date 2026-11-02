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
// Explicit separate entry point: old addons cannot silently treat a requested
// Chrome renderer PID as the Node driver's own PID.
static napi_value instructions_for_pid(napi_env env, napi_callback_info info) {
    size_t argc=1;napi_value argv[1];int32_t pid;
    if(napi_get_cb_info(env,info,&argc,argv,0,0)!=napi_ok || argc!=1 ||
       napi_get_value_int32(env,argv[0],&pid)!=napi_ok || pid<=0) {
        napi_throw_type_error(env,0,"expected a positive process PID");return 0;
    }
    struct rusage_info_v4 usage;
    if(proc_pid_rusage(pid,RUSAGE_INFO_V4,(rusage_info_t *)&usage)) {
        napi_throw_error(env,0,"proc_pid_rusage for renderer failed");return 0;
    }
    napi_value result;napi_create_double(env,(double)usage.ri_instructions,&result);return result;
}
static napi_value init(napi_env env,napi_value exports) {
    napi_value fn;napi_create_function(env,"instructions",NAPI_AUTO_LENGTH,instructions,0,&fn);
    napi_set_named_property(env,exports,"instructions",fn);
    napi_create_function(env,"instructionsForPid",NAPI_AUTO_LENGTH,instructions_for_pid,0,&fn);
    napi_set_named_property(env,exports,"instructionsForPid",fn);return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME,init)
