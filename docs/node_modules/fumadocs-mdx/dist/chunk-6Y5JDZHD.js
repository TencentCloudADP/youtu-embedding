// src/utils/import-formatter.ts
import path from "path";
function getImportCode(info) {
  const specifier = JSON.stringify(info.specifier);
  if (info.type === "default") return `import ${info.name} from ${specifier}`;
  if (info.type === "namespace")
    return `import * as ${info.name} from ${specifier}`;
  if (info.type === "named") {
    const names = info.names.map(
      (name) => Array.isArray(name) ? `${name[0]} as ${name[1]}` : name
    );
    return `import { ${names.join(", ")} } from ${specifier}`;
  }
  return `import ${specifier}`;
}
function toImportPath(file, config) {
  const ext = path.extname(file);
  let filename;
  if (ext === ".ts" && config.jsExtension) {
    filename = file.substring(0, file.length - ext.length) + ".js";
  } else if (ext === ".ts") {
    filename = file.substring(0, file.length - ext.length);
  } else {
    filename = file;
  }
  let importPath;
  if ("relativeTo" in config) {
    importPath = path.relative(config.relativeTo, filename);
    if (!path.isAbsolute(importPath) && !importPath.startsWith(".")) {
      importPath = `./${importPath}`;
    }
  } else {
    importPath = path.resolve(filename);
  }
  return importPath.replaceAll(path.sep, "/");
}
function ident(code, tab = 1) {
  return code.split("\n").map((v) => "  ".repeat(tab) + v).join("\n");
}

// src/utils/collections.ts
function getSupportedFormats(collection) {
  return {
    doc: ["mdx", "md"],
    meta: ["json", "yaml"]
  }[collection.type];
}
function getGlobPatterns(collection) {
  if (collection.files) return collection.files;
  return [`**/*.{${getSupportedFormats(collection).join(",")}}`];
}
function isFileSupported(filePath, collection) {
  for (const format of getSupportedFormats(collection)) {
    if (filePath.endsWith(`.${format}`)) return true;
  }
  return false;
}

export {
  getImportCode,
  toImportPath,
  ident,
  getGlobPatterns,
  isFileSupported
};
