// Whether an element is drawn faded: it or an ancestor carries the dim class or an
// opacity below 1. A switched-off item fades its content this way, never its buttons.
export function faded(el) {
  for (; el; el = el.parentElement) {
    if (el.classList.contains("z-dim") || (el.style.opacity && +el.style.opacity < 1)) return true;
  }
  return false;
}
