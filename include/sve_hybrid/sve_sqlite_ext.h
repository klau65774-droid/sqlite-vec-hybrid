#ifndef SVE_SQLITE_EXT_H
#define SVE_SQLITE_EXT_H

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

#ifdef SVE_HYBRID_STATIC
#define SVE_HYBRID_API
#else
#ifdef _WIN32
#define SVE_HYBRID_API __declspec(dllexport)
#else
#define SVE_HYBRID_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Registers `sve0` virtual table + all `sve_*` scalar functions (SQLITE_CORE) */
SVE_HYBRID_API int sve_hybrid_register(sqlite3 *db, char **pzErrMsg);

/* Standard SQLite extension initialization entry point */
SVE_HYBRID_API int sqlite3_svehybrid_init(sqlite3 *db, char **pzErrMsg,
                                          const sqlite3_api_routines *pApi);
/* Entry point inferred by SQLite from the vec_hybrid library filename. */
SVE_HYBRID_API int sqlite3_vechybrid_init(sqlite3 *db, char **pzErrMsg,
                                          const sqlite3_api_routines *pApi);

#ifdef __cplusplus
}
#endif

#endif /* SVE_SQLITE_EXT_H */
