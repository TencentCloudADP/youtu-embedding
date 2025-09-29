// src/source/path.ts
function basename(path, ext) {
  const idx = path.lastIndexOf("/");
  return path.substring(
    idx === -1 ? 0 : idx + 1,
    ext ? path.length - ext.length : path.length
  );
}
function extname(path) {
  const dotIdx = path.lastIndexOf(".");
  if (dotIdx !== -1) {
    return path.substring(dotIdx);
  }
  return "";
}
function dirname(path) {
  return path.split("/").slice(0, -1).join("/");
}
function parseFilePath(path) {
  const ext = extname(path);
  const name = basename(path, ext);
  const dir = dirname(path);
  return {
    dirname: dir,
    name,
    ext,
    path,
    get flattenedPath() {
      return [dir, name].filter((p) => p.length > 0).join("/");
    }
  };
}
function parseFolderPath(path) {
  return {
    dirname: dirname(path),
    name: basename(path),
    path
  };
}
function splitPath(path) {
  return path.split("/").filter((p) => p.length > 0);
}
function joinPath(...paths) {
  const out = [];
  const parsed = paths.flatMap(splitPath);
  for (const seg of parsed) {
    switch (seg) {
      case "..":
        out.pop();
        break;
      case ".":
        break;
      default:
        out.push(seg);
    }
  }
  return out.join("/");
}
function slash(path) {
  const isExtendedLengthPath = path.startsWith("\\\\?\\");
  if (isExtendedLengthPath) {
    return path;
  }
  return path.replaceAll("\\", "/");
}

export {
  basename,
  extname,
  dirname,
  parseFilePath,
  parseFolderPath,
  splitPath,
  joinPath,
  slash
};
