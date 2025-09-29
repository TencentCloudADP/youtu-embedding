"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BoolNode = void 0;
class BoolNode {
    true;
    false;
    constructor() {
        this.true = new Set();
        this.false = new Set();
    }
    insert(value, bool) {
        if (bool) {
            this.true.add(value);
        }
        else {
            this.false.add(value);
        }
    }
    delete(value, bool) {
        if (bool) {
            this.true.delete(value);
        }
        else {
            this.false.delete(value);
        }
    }
    getSize() {
        return this.true.size + this.false.size;
    }
    toJSON() {
        return {
            true: Array.from(this.true),
            false: Array.from(this.false)
        };
    }
    static fromJSON(json) {
        const node = new BoolNode();
        node.true = new Set(json.true);
        node.false = new Set(json.false);
        return node;
    }
}
exports.BoolNode = BoolNode;
//# sourceMappingURL=bool.js.map