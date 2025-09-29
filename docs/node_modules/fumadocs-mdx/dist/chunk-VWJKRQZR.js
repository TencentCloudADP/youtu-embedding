// src/utils/fuma-matter.ts
import { load } from "js-yaml";
var regex = /^---\r?\n(.+?)\r?\n---\r?\n/s;
function fumaMatter(input) {
  const output = { matter: "", data: {}, content: input };
  const match = regex.exec(input);
  if (!match) {
    return output;
  }
  output.matter = match[0];
  output.content = input.slice(match[0].length);
  const loaded = load(match[1]);
  output.data = loaded ?? {};
  return output;
}

export {
  fumaMatter
};
