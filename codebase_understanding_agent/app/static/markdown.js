(() => {
  "use strict";

  const INLINE_MARKERS = new Set(["`", "[", "*", "~", "<", "\\"]);

  function appendText(parent, value) {
    if (value) {
      parent.append(document.createTextNode(value));
    }
  }

  function safeLinkTarget(rawTarget) {
    const target = String(rawTarget || "").trim();
    if (!target) {
      return null;
    }
    if (target.startsWith("#") || target.startsWith("/")) {
      return target;
    }
    if (!/^(https?:|mailto:)/i.test(target)) {
      return null;
    }
    try {
      const parsed = new URL(target, window.location.href);
      return ["http:", "https:", "mailto:"].includes(parsed.protocol) ? parsed.href : null;
    } catch (_error) {
      return null;
    }
  }

  function findClosingMarker(text, marker, start) {
    let cursor = start;
    while (cursor < text.length) {
      const match = text.indexOf(marker, cursor);
      if (match === -1) {
        return -1;
      }
      if (match === 0 || text[match - 1] !== "\\") {
        return match;
      }
      cursor = match + marker.length;
    }
    return -1;
  }

  function appendInline(parent, source, depth = 0) {
    const text = String(source || "");
    if (depth > 8) {
      appendText(parent, text);
      return;
    }

    let position = 0;
    while (position < text.length) {
      const character = text[position];

      if (character === "\\" && position + 1 < text.length) {
        appendText(parent, text[position + 1]);
        position += 2;
        continue;
      }

      if (character === "`") {
        const markerLength = text[position + 1] === "`" ? 2 : 1;
        const marker = "`".repeat(markerLength);
        const closing = findClosingMarker(text, marker, position + markerLength);
        if (closing !== -1) {
          const code = document.createElement("code");
          code.textContent = text.slice(position + markerLength, closing).trim();
          parent.append(code);
          position = closing + markerLength;
          continue;
        }
      }

      if (text.startsWith("**", position)) {
        const closing = findClosingMarker(text, "**", position + 2);
        if (closing > position + 2) {
          const strong = document.createElement("strong");
          appendInline(strong, text.slice(position + 2, closing), depth + 1);
          parent.append(strong);
          position = closing + 2;
          continue;
        }
      }

      if (text.startsWith("~~", position)) {
        const closing = findClosingMarker(text, "~~", position + 2);
        if (closing > position + 2) {
          const deleted = document.createElement("del");
          appendInline(deleted, text.slice(position + 2, closing), depth + 1);
          parent.append(deleted);
          position = closing + 2;
          continue;
        }
      }

      if (character === "*" && text[position + 1] !== "*") {
        const closing = findClosingMarker(text, "*", position + 1);
        if (closing > position + 1) {
          const emphasis = document.createElement("em");
          appendInline(emphasis, text.slice(position + 1, closing), depth + 1);
          parent.append(emphasis);
          position = closing + 1;
          continue;
        }
      }

      if (character === "[") {
        const labelEnd = findClosingMarker(text, "]", position + 1);
        if (labelEnd !== -1 && text[labelEnd + 1] === "(") {
          const targetEnd = findClosingMarker(text, ")", labelEnd + 2);
          if (targetEnd !== -1) {
            const rawTarget = text.slice(labelEnd + 2, targetEnd).trim().split(/\s+/)[0];
            const target = safeLinkTarget(rawTarget);
            if (target) {
              const link = document.createElement("a");
              link.href = target;
              appendInline(link, text.slice(position + 1, labelEnd), depth + 1);
              if (!target.startsWith("#") && !target.startsWith("/")) {
                link.target = "_blank";
                link.rel = "noopener noreferrer";
              }
              parent.append(link);
              position = targetEnd + 1;
              continue;
            }
          }
        }
      }

      if (character === "<") {
        const closing = text.indexOf(">", position + 1);
        if (closing !== -1) {
          const target = safeLinkTarget(text.slice(position + 1, closing));
          if (target && !target.startsWith("#") && !target.startsWith("/")) {
            const link = document.createElement("a");
            link.href = target;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = text.slice(position + 1, closing);
            parent.append(link);
            position = closing + 1;
            continue;
          }
        }
      }

      let next = position + 1;
      while (next < text.length && !INLINE_MARKERS.has(text[next])) {
        next += 1;
      }
      appendText(parent, text.slice(position, next));
      position = next;
    }
  }

  function splitTableRow(line) {
    let value = line.trim();
    if (value.startsWith("|")) value = value.slice(1);
    if (value.endsWith("|")) value = value.slice(0, -1);

    const cells = [];
    let cell = "";
    let escaped = false;
    for (const character of value) {
      if (escaped) {
        cell += character;
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === "|") {
        cells.push(cell.trim());
        cell = "";
      } else {
        cell += character;
      }
    }
    cells.push(cell.trim());
    return cells;
  }

  function tableAlignments(line) {
    const cells = splitTableRow(line);
    if (!cells.length || !cells.every((cell) => /^:?-{3,}:?$/.test(cell))) {
      return null;
    }
    return cells.map((cell) => {
      if (cell.startsWith(":") && cell.endsWith(":")) return "center";
      if (cell.endsWith(":")) return "right";
      return "left";
    });
  }

  function listMatch(line) {
    return line.match(/^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/);
  }

  function isFence(line) {
    return /^\s{0,3}```/.test(line);
  }

  function isBlockStart(lines, index) {
    const line = lines[index] || "";
    const next = lines[index + 1] || "";
    return (
      !line.trim() ||
      isFence(line) ||
      /^\s{0,3}#{1,6}\s+/.test(line) ||
      /^\s{0,3}>/.test(line) ||
      /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line) ||
      Boolean(listMatch(line)) ||
      (line.includes("|") && Boolean(tableAlignments(next)))
    );
  }

  function appendCodeBlock(root, language, content) {
    const wrapper = document.createElement("div");
    wrapper.className = "markdown-code-block";
    const meta = document.createElement("div");
    meta.className = "markdown-code-meta";
    meta.textContent = language || "code";
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    if (language) {
      code.className = `language-${language.replace(/[^a-zA-Z0-9_+#.-]/g, "")}`;
    }
    code.textContent = content;
    pre.append(code);
    wrapper.append(meta, pre);
    root.append(wrapper);
  }

  function renderMarkdown(source) {
    const root = document.createElement("div");
    root.className = "markdown-body";
    const lines = String(source || "").replace(/\r\n?/g, "\n").split("\n");
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = line.match(/^\s{0,3}```\s*([^\s`]*)\s*$/);
      if (fence) {
        const codeLines = [];
        index += 1;
        while (index < lines.length && !/^\s{0,3}```\s*$/.test(lines[index])) {
          codeLines.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        appendCodeBlock(root, fence[1], codeLines.join("\n"));
        continue;
      }

      const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
      if (heading) {
        const element = document.createElement(`h${heading[1].length}`);
        appendInline(element, heading[2]);
        root.append(element);
        index += 1;
        continue;
      }

      if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
        root.append(document.createElement("hr"));
        index += 1;
        continue;
      }

      if (/^\s{0,3}>/.test(line)) {
        const quoteLines = [];
        while (index < lines.length && /^\s{0,3}>/.test(lines[index])) {
          quoteLines.push(lines[index].replace(/^\s{0,3}>\s?/, ""));
          index += 1;
        }
        const quote = document.createElement("blockquote");
        quote.append(...renderMarkdown(quoteLines.join("\n")).childNodes);
        root.append(quote);
        continue;
      }

      const alignment = index + 1 < lines.length ? tableAlignments(lines[index + 1]) : null;
      if (line.includes("|") && alignment) {
        const headers = splitTableRow(line);
        const table = document.createElement("table");
        const head = document.createElement("thead");
        const headRow = document.createElement("tr");
        headers.forEach((header, cellIndex) => {
          const cell = document.createElement("th");
          cell.style.textAlign = alignment[cellIndex] || "left";
          appendInline(cell, header);
          headRow.append(cell);
        });
        head.append(headRow);
        table.append(head);
        index += 2;

        const body = document.createElement("tbody");
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          const row = document.createElement("tr");
          splitTableRow(lines[index]).forEach((value, cellIndex) => {
            const cell = document.createElement("td");
            cell.style.textAlign = alignment[cellIndex] || "left";
            appendInline(cell, value);
            row.append(cell);
          });
          body.append(row);
          index += 1;
        }
        table.append(body);
        const frame = document.createElement("div");
        frame.className = "markdown-table-frame";
        frame.append(table);
        root.append(frame);
        continue;
      }

      const firstListItem = listMatch(line);
      if (firstListItem) {
        const ordered = /^\d/.test(firstListItem[1]);
        const list = document.createElement(ordered ? "ol" : "ul");
        while (index < lines.length) {
          const item = listMatch(lines[index]);
          if (!item || /^\d/.test(item[1]) !== ordered) break;
          const entry = document.createElement("li");
          appendInline(entry, item[2]);
          list.append(entry);
          index += 1;
        }
        root.append(list);
        continue;
      }

      const paragraphLines = [line];
      index += 1;
      while (index < lines.length && !isBlockStart(lines, index)) {
        paragraphLines.push(lines[index]);
        index += 1;
      }
      const paragraph = document.createElement("p");
      paragraphLines.forEach((paragraphLine, lineIndex) => {
        const hardBreak = /\s{2}$/.test(paragraphLine);
        appendInline(paragraph, paragraphLine.replace(/\s+$/, ""));
        if (lineIndex < paragraphLines.length - 1) {
          paragraph.append(hardBreak ? document.createElement("br") : document.createTextNode(" "));
        }
      });
      root.append(paragraph);
    }

    return root;
  }

  window.CodeAgentMarkdown = Object.freeze({ render: renderMarkdown });
})();
