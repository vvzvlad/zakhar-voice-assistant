// Helpers for reading pydantic-emitted JSON Schema. Pure functions, no React.

// Follow a "#/$defs/Name" $ref inside `root`.
export function deref(node, root) {
  let seen = 0;
  while (node && node.$ref && seen < 10) {
    const ref = node.$ref;
    const m = /^#\/\$defs\/(.+)$/.exec(ref);
    const defs = (root && root.$defs) || {};
    const target = m ? defs[m[1]] : null;
    if (!target) break;
    // Merge sibling keys (e.g. a `default` next to a $ref) over the target.
    const { $ref, ...rest } = node;
    node = { ...target, ...rest };
    seen++;
  }
  return node || {};
}

// Fully resolve a property node: follow $ref and collapse single-branch
// allOf/anyOf/oneOf so callers can read `enum`, `type`, `minimum`, etc. directly.
export function resolve(node, root) {
  let n = deref(node, root);
  for (const key of ["allOf", "anyOf", "oneOf"]) {
    if (Array.isArray(n[key])) {
      // Pick the first non-null branch (pydantic often emits [{...}, {type:null}]).
      const branches = n[key].map((b) => deref(b, root)).filter((b) => b && b.type !== "null");
      const pick = branches[0] || {};
      const { [key]: _drop, ...rest } = n;
      n = { ...pick, ...rest };
      n = deref(n, root);
    }
  }
  return n;
}

// Extract an enum from a (possibly wrapped) property node.
export function enumOf(node, root) {
  const n = resolve(node, root);
  if (Array.isArray(n.enum)) return n.enum;
  // anyOf/oneOf where one branch carries the enum.
  for (const key of ["anyOf", "oneOf", "allOf"]) {
    if (Array.isArray(node[key])) {
      for (const b of node[key]) {
        const r = resolve(b, root);
        if (Array.isArray(r.enum)) return r.enum;
      }
    }
  }
  return null;
}

const KEYISH = /(api_key|token|psk|secret|key|password|passwd|pwd)/i;
export const isSecret = (name) => KEYISH.test(name);

// Title-case a snake_case field name as a fallback label.
export function humanize(name) {
  return name
    .split("_")
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}
