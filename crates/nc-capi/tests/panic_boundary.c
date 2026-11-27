/*
 * The C ABI's panic guard, seen from C: a panic inside the library must come
 * back as NC_ERROR_INTERNAL and leave the process running, not abort it.
 * Built by tools/ci/capi_check.sh against a `capi`-profile library with the
 * `panic-probe` feature, which exports nc_debug_panic (not in the header).
 */
#include <stdio.h>

#include "nanocodec.h"

nc_status nc_debug_panic(void);

int main(void) {
    for (int i = 0; i < 2; i++) {
        nc_status status = nc_debug_panic();
        if (status != NC_ERROR_INTERNAL) {
            printf("panic %d returned %d, not NC_ERROR_INTERNAL\n", i, (int)status);
            return 1;
        }
    }
    nc_decoder *decoder = NULL;
    if (nc_decoder_create(NC_CODEC_H264, &decoder) != NC_OK) {
        printf("the library did not work after a caught panic\n");
        return 1;
    }
    nc_decoder_destroy(decoder);
    printf("panics returned NC_ERROR_INTERNAL; the library still works\n");
    return 0;
}
