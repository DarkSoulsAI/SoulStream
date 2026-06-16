const root = document.documentElement;
const buttons = Array.from(document.querySelectorAll("[data-view-button]"));
const storageKey = "soulstream-docs-language-view";

function readSavedView() {
  try {
    return localStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function saveView(view) {
  try {
    localStorage.setItem(storageKey, view);
  } catch {
    // The language toggle should still work when storage is blocked.
  }
}

function setView(view) {
  root.dataset.view = view;
  buttons.forEach((button) => {
    const active = button.dataset.viewButton === view;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  saveView(view);
}

buttons.forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.viewButton));
});

const saved = readSavedView();
if (["both", "zh", "en"].includes(saved)) {
  setView(saved);
} else {
  setView("both");
}
