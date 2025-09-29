"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "transpileConfig", {
    enumerable: true,
    get: function() {
        return transpileConfig;
    }
});
const _nodepath = require("node:path");
const _promises = require("node:fs/promises");
const _requirehook = require("./require-hook");
const _log = require("../output/log");
const _installdependencies = require("../../lib/install-dependencies");
function resolveSWCOptions(cwd, compilerOptions) {
    const resolvedBaseUrl = (0, _nodepath.resolve)(cwd, compilerOptions.baseUrl ?? '.');
    return {
        jsc: {
            target: 'es5',
            parser: {
                syntax: 'typescript'
            },
            paths: compilerOptions.paths,
            baseUrl: resolvedBaseUrl
        },
        module: {
            type: 'commonjs'
        },
        isModule: 'unknown'
    };
}
// Ported from next/src/lib/verify-typescript-setup.ts
// Although this overlaps with the later `verifyTypeScriptSetup`,
// it is acceptable since the time difference in the worst case is trivial,
// as we are only preparing to install the dependencies once more.
async function verifyTypeScriptSetup(cwd, configFileName) {
    try {
        // Quick module check.
        require.resolve('typescript', {
            paths: [
                cwd
            ]
        });
    } catch (error) {
        if (error && typeof error === 'object' && 'code' in error && error.code === 'MODULE_NOT_FOUND') {
            (0, _log.warn)(`Installing TypeScript as it was not found while loading "${configFileName}".`);
            await (0, _installdependencies.installDependencies)(cwd, [
                {
                    pkg: 'typescript'
                }
            ], true).catch((err)=>{
                if (err && typeof err === 'object' && 'command' in err) {
                    console.error(`Failed to install TypeScript, please install it manually to continue:\n` + err.command + '\n');
                }
                throw err;
            });
        }
    }
}
async function getTsConfig(cwd) {
    const ts = require(require.resolve('typescript', {
        paths: [
            cwd
        ]
    }));
    // NOTE: This doesn't fully cover the edge case for setting
    // "typescript.tsconfigPath" in next config which is currently
    // a restriction.
    const tsConfigPath = ts.findConfigFile(cwd, ts.sys.fileExists, 'tsconfig.json');
    if (!tsConfigPath) {
        // It is ok to not return ts.getDefaultCompilerOptions() because
        // we are only lookfing for paths and baseUrl from tsConfig.
        return {};
    }
    const configFile = ts.readConfigFile(tsConfigPath, ts.sys.readFile);
    const parsedCommandLine = ts.parseJsonConfigFileContent(configFile.config, ts.sys, cwd);
    return parsedCommandLine.options;
}
async function transpileConfig({ nextConfigPath, configFileName, cwd }) {
    let hasRequire = false;
    try {
        // Ensure TypeScript is installed to use the API.
        await verifyTypeScriptSetup(cwd, configFileName);
        const compilerOptions = await getTsConfig(cwd);
        const swcOptions = resolveSWCOptions(cwd, compilerOptions);
        const nextConfigString = await (0, _promises.readFile)(nextConfigPath, 'utf8');
        // lazy require swc since it loads React before even setting NODE_ENV
        // resulting loading Development React on Production
        const { transform } = require('../swc');
        const { code } = await transform(nextConfigString, swcOptions);
        // register require hook only if require exists
        if (code.includes('require(')) {
            (0, _requirehook.registerHook)(swcOptions);
            hasRequire = true;
        }
        // filename & extension don't matter here
        return (0, _requirehook.requireFromString)(code, (0, _nodepath.resolve)(cwd, 'next.config.compiled.js'));
    } catch (error) {
        throw error;
    } finally{
        if (hasRequire) {
            (0, _requirehook.deregisterHook)();
        }
    }
}

//# sourceMappingURL=transpile-config.js.map