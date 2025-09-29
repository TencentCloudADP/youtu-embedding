"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AVAILABLE_PLUGIN_HOOKS = void 0;
exports.getAllPluginsByHook = getAllPluginsByHook;
const errors_js_1 = require("../errors.js");
exports.AVAILABLE_PLUGIN_HOOKS = [
    'beforeInsert',
    'afterInsert',
    'beforeRemove',
    'afterRemove',
    'beforeUpdate',
    'afterUpdate',
    'beforeUpsert',
    'afterUpsert',
    'beforeSearch',
    'afterSearch',
    'beforeInsertMultiple',
    'afterInsertMultiple',
    'beforeRemoveMultiple',
    'afterRemoveMultiple',
    'beforeUpdateMultiple',
    'afterUpdateMultiple',
    'beforeUpsertMultiple',
    'afterUpsertMultiple',
    'beforeLoad',
    'afterLoad',
    'afterCreate'
];
function getAllPluginsByHook(orama, hook) {
    const pluginsToRun = [];
    const pluginsLength = orama.plugins?.length;
    if (!pluginsLength) {
        return pluginsToRun;
    }
    for (let i = 0; i < pluginsLength; i++) {
        try {
            const plugin = orama.plugins[i];
            if (typeof plugin[hook] === 'function') {
                pluginsToRun.push(plugin[hook]);
            }
        }
        catch (error) {
            console.error('Caught error in getAllPluginsByHook:', error);
            throw (0, errors_js_1.createError)('PLUGIN_CRASHED');
        }
    }
    return pluginsToRun;
}
//# sourceMappingURL=plugins.js.map