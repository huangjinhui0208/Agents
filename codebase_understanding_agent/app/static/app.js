(() => {
  "use strict";

  const state = {
    config: null,
    repository: null,
    revision: null,
    summary: null,
    sessionId: null,
    busy: false,
    browsePath: "",
    browseParent: null,
  };

  const elements = {
    runtimeDot: document.querySelector("#runtime-dot"),
    runtimeStatus: document.querySelector("#runtime-status"),
    modelName: document.querySelector("#model-name"),
    repositoryForm: document.querySelector("#repository-form"),
    repositoryPath: document.querySelector("#repository-path"),
    loadRepository: document.querySelector("#load-repository"),
    browseButton: document.querySelector("#browse-button"),
    repositoryCard: document.querySelector("#repository-card"),
    repositoryName: document.querySelector("#repository-name"),
    repositoryMeta: document.querySelector("#repository-meta"),
    activeRepository: document.querySelector("#active-repository"),
    revisionValue: document.querySelector("#revision-value"),
    fileCountValue: document.querySelector("#file-count-value"),
    sessionValue: document.querySelector("#session-value"),
    limitsNote: document.querySelector("#limits-note"),
    resetSession: document.querySelector("#reset-session"),
    messageLog: document.querySelector("#message-log"),
    emptyState: document.querySelector("#empty-state"),
    chatComposer: document.querySelector("#chat-composer"),
    questionInput: document.querySelector("#question-input"),
    composerHint: document.querySelector("#composer-hint"),
    sendButton: document.querySelector("#send-button"),
    repositoryBrowser: document.querySelector("#repository-browser"),
    browserPath: document.querySelector("#browser-path"),
    browserError: document.querySelector("#browser-error"),
    parentDirectory: document.querySelector("#parent-directory"),
    directoryList: document.querySelector("#directory-list"),
    selectCurrentDirectory: document.querySelector("#select-current-directory"),
    toastRegion: document.querySelector("#toast-region"),
  };

  function createElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }
    if (text !== undefined && text !== null) {
      element.textContent = String(text);
    }
    return element;
  }

  function valueAsText(value, fallback = "") {
    if (value === undefined || value === null || value === "") {
      return fallback;
    }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    try {
      return JSON.stringify(value);
    } catch (_error) {
      return fallback;
    }
  }

  function renderAssistantAnswer(value) {
    const answer = valueAsText(value, "Agent 已结束本轮，但没有返回可显示的答案。");
    if (window.CodeAgentMarkdown?.render) {
      return window.CodeAgentMarkdown.render(answer);
    }
    return createElement("p", "answer-text", answer);
  }

  function shortValue(value, length = 12) {
    const text = valueAsText(value, "—");
    return text.length > length ? `${text.slice(0, length)}…` : text;
  }

  function pathBasename(path) {
    const parts = valueAsText(path).split(/[\\/]/).filter(Boolean);
    return parts.at(-1) || valueAsText(path, "未命名仓库");
  }

  function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat("zh-CN").format(number) : "—";
  }

  function formatEndpoint(baseUrl) {
    if (!baseUrl) {
      return "本机服务";
    }
    try {
      const url = new URL(baseUrl);
      const isLocal = ["127.0.0.1", "localhost", "::1"].includes(url.hostname);
      const port = url.port ? `:${url.port}` : "";
      return isLocal ? `本机${port}` : `${url.hostname}${port}`;
    } catch (_error) {
      return valueAsText(baseUrl);
    }
  }

  function normalizeStatus(status) {
    if (status && typeof status === "object") {
      status = status.state ?? status.status ?? status.ollama ?? status.available;
    }
    return valueAsText(status, "unknown").toLowerCase();
  }

  function isHealthyStatus(status) {
    return ["ok", "ready", "online", "available", "connected", "healthy", "true"].some(
      (token) => status === token || status.includes(token),
    );
  }

  function isUnhealthyStatus(status) {
    return ["error", "offline", "unavailable", "disconnected", "false", "failed"].some(
      (token) => status === token || status.includes(token),
    );
  }

  async function requestJson(url, options = {}) {
    const requestOptions = {
      method: options.method || "GET",
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    };

    if (Object.prototype.hasOwnProperty.call(options, "body")) {
      requestOptions.headers["Content-Type"] = "application/json";
      requestOptions.body = JSON.stringify(options.body);
    }

    let response;
    try {
      response = await fetch(url, requestOptions);
    } catch (_error) {
      throw new Error("无法连接本地 Agent 服务。请确认服务仍在运行。");
    }

    const responseText = await response.text();
    let data = {};
    if (responseText) {
      try {
        data = JSON.parse(responseText);
      } catch (_error) {
        if (!response.ok) {
          throw new Error(`服务返回了无法解析的错误（HTTP ${response.status}）。`);
        }
        throw new Error("服务返回内容不是有效的 JSON。");
      }
    }

    const serverError = data && data.error;
    if (!response.ok || serverError) {
      const message =
        (serverError && typeof serverError === "object" && serverError.message) ||
        (typeof serverError === "string" && serverError) ||
        data.message ||
        `请求失败（HTTP ${response.status}）。`;
      const error = new Error(valueAsText(message, "请求失败。"));
      error.code = serverError && typeof serverError === "object" ? serverError.code : undefined;
      error.status = response.status;
      throw error;
    }

    return data;
  }

  function showToast(message, variant = "info") {
    const toast = createElement("div", `toast is-${variant}`, message);
    elements.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), 4600);
  }

  function setRuntimeStatus(status, label) {
    elements.runtimeDot.className = `status-dot is-${status}`;
    elements.runtimeStatus.textContent = label;
  }

  function setBusy(busy, label = "") {
    state.busy = busy;
    elements.messageLog.setAttribute("aria-busy", String(busy));
    elements.repositoryPath.disabled = busy;
    elements.loadRepository.disabled = busy;
    elements.browseButton.disabled = busy;
    elements.resetSession.disabled = busy || !state.sessionId;
    elements.loadRepository.textContent = busy && label === "repository" ? "正在载入…" : "载入此仓库";
    elements.sendButton.querySelector("span:first-child").textContent =
      busy && label === "chat" ? "正在检索…" : "发送问题";
    updateComposerState();
  }

  function updateComposerState() {
    const hasSession = Boolean(state.sessionId);
    const hasQuestion = Boolean(elements.questionInput.value.trim());
    elements.questionInput.disabled = !hasSession;
    elements.sendButton.disabled = !hasSession || !hasQuestion || state.busy;
    elements.composerHint.textContent = hasSession
      ? "Enter 发送 · Shift + Enter 换行"
      : "载入仓库后可以提问";
  }

  function autoResizeQuestion() {
    elements.questionInput.style.height = "auto";
    elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 190)}px`;
  }

  function scrollMessagesToEnd() {
    window.requestAnimationFrame(() => {
      elements.messageLog.scrollTop = elements.messageLog.scrollHeight;
    });
  }

  function updateEmptyState() {
    const title = elements.emptyState.querySelector("h3");
    const description = elements.emptyState.querySelector("p");
    if (state.repository) {
      title.textContent = `${valueAsText(state.repository.name, "仓库")} 已就绪`;
      description.textContent =
        "从一个具体的问题开始。Agent 会按需定位文件、阅读相关行，并在回答下方保留源码证据轨迹。";
    } else {
      title.textContent = "先载入仓库，再从一个具体问题开始";
      description.textContent =
        "适合询问入口、模块职责、调用路径和实现依据。每次回答会把检索过程整理成可折叠的证据轨迹。";
    }
  }

  function resetMessageView() {
    updateEmptyState();
    elements.messageLog.replaceChildren(elements.emptyState);
    scrollMessagesToEnd();
  }

  function updateRepositoryView() {
    const repository = state.repository;
    if (!repository) {
      elements.repositoryCard.classList.add("is-empty");
      elements.repositoryName.textContent = "尚未选择仓库";
      elements.repositoryMeta.textContent = "载入后可开始提问";
      elements.activeRepository.textContent = "未连接仓库";
      elements.revisionValue.textContent = "—";
      elements.fileCountValue.textContent = "—";
      elements.sessionValue.textContent = "—";
      return;
    }

    const path = valueAsText(repository.path);
    const name = valueAsText(repository.name, pathBasename(path));
    elements.repositoryCard.classList.remove("is-empty");
    elements.repositoryName.textContent = name;
    elements.repositoryMeta.textContent = path;
    elements.repositoryMeta.title = path;
    elements.activeRepository.textContent = name;
    elements.activeRepository.title = path;
    elements.revisionValue.textContent = shortValue(state.revision);
    elements.revisionValue.title = valueAsText(state.revision, "");
    elements.fileCountValue.textContent = formatNumber(state.summary?.file_count);
    elements.sessionValue.textContent = shortValue(state.sessionId, 10);
    elements.sessionValue.title = valueAsText(state.sessionId, "");
  }

  async function loadConfig() {
    try {
      const config = await requestJson("/api/config");
      state.config = config;
      elements.modelName.textContent = valueAsText(config.model, "本地模型");
      elements.modelName.title = valueAsText(config.model, "");

      const status = normalizeStatus(config.status);
      const endpoint = formatEndpoint(config.base_url);
      const provider = valueAsText(config.provider, "model");
      if (isHealthyStatus(status)) {
        setRuntimeStatus("online", `${provider} 已连接 · ${endpoint}`);
      } else if (isUnhealthyStatus(status)) {
        setRuntimeStatus("offline", `${provider} 未就绪 · ${endpoint}`);
      } else {
        setRuntimeStatus("loading", `${provider} · ${endpoint}`);
      }

      if (config.default_browse_path && !elements.repositoryPath.value) {
        elements.repositoryPath.value = valueAsText(config.default_browse_path);
      }

      const maxSteps = config.limits?.max_steps;
      const maxLines = config.limits?.max_file_lines;
      const facts = [];
      if (maxSteps !== undefined) {
        facts.push(`每次最多 ${maxSteps} 步`);
      }
      if (maxLines !== undefined) {
        facts.push(`单次读取 ${maxLines} 行`);
      }
      elements.limitsNote.textContent = facts.length
        ? `运行上限：${facts.join(" · ")}`
        : "运行上限由本地服务控制。";
    } catch (error) {
      setRuntimeStatus("offline", "本地服务未就绪");
      elements.modelName.textContent = "模型状态未知";
      elements.limitsNote.textContent = "无法读取运行限制。";
      showToast(error.message, "error");
    }
  }

  async function selectRepository(path) {
    const selectedPath = valueAsText(path).trim();
    if (!selectedPath) {
      showToast("请输入仓库绝对路径，或打开目录浏览器选择。", "error");
      elements.repositoryPath.focus();
      return false;
    }

    setBusy(true, "repository");
    try {
      const data = await requestJson("/api/repository", {
        method: "POST",
        body: { path: selectedPath },
      });
      if (!data.session_id || !data.repository) {
        throw new Error("服务未返回有效的仓库会话。");
      }

      state.sessionId = valueAsText(data.session_id);
      state.repository = data.repository;
      state.revision = data.revision ?? null;
      state.summary = data.summary ?? {};
      elements.repositoryPath.value = valueAsText(data.repository.path, selectedPath);
      updateRepositoryView();
      resetMessageView();

      if (elements.repositoryBrowser.open) {
        elements.repositoryBrowser.close();
      }
      showToast(`已载入 ${valueAsText(data.repository.name, pathBasename(selectedPath))}`, "success");
      window.requestAnimationFrame(() => elements.questionInput.focus());
      return true;
    } catch (error) {
      showToast(error.message, "error");
      return false;
    } finally {
      setBusy(false);
    }
  }

  function appendUserMessage(question) {
    if (elements.emptyState.isConnected) {
      elements.emptyState.remove();
    }
    const article = createElement("article", "message-entry is-user");
    const label = createElement("div", "message-label", "你");
    const body = createElement("div", "message-body");
    body.append(createElement("p", "answer-text", question));
    article.append(label, body);
    elements.messageLog.append(article);
    scrollMessagesToEnd();
  }

  function appendPendingMessage() {
    const article = createElement("article", "message-entry is-pending");
    article.dataset.pending = "true";
    article.append(createElement("div", "message-label", "Agent · 检索中"));

    const body = createElement("div", "message-body");
    const line = createElement("div", "pending-line");
    const activity = createElement("span", "activity-mark");
    activity.setAttribute("aria-hidden", "true");
    activity.append(createElement("i"), createElement("i"), createElement("i"));
    line.append(activity, createElement("span", "", "正在定位相关源码并核对证据…"));
    body.append(line);
    article.append(body);
    elements.messageLog.append(article);
    scrollMessagesToEnd();
    return article;
  }

  function citationLabel(citation) {
    const path = valueAsText(citation.path, "未知路径");
    const start = Number(citation.start_line);
    const end = Number(citation.end_line);
    if (Number.isFinite(start) && Number.isFinite(end) && end !== start) {
      return `${path}:L${start}–L${end}`;
    }
    if (Number.isFinite(start)) {
      return `${path}:L${start}`;
    }
    return path;
  }

  function renderEvidenceTrace(trace, citations) {
    const safeTrace = Array.isArray(trace) ? trace : [];
    const safeCitations = Array.isArray(citations) ? citations : [];
    const details = createElement("details", "evidence-trace");
    const summary = createElement("summary");
    summary.append(
      createElement("span", "", "源码证据轨迹"),
      createElement(
        "span",
        "trace-count",
        `${safeTrace.length} 次工具调用 · ${safeCitations.length} 个引用`,
      ),
    );
    details.append(summary);

    const content = createElement("div", "trace-content");
    if (safeTrace.length) {
      content.append(createElement("h3", "trace-heading", "工具调用轨迹"));
      const timeline = createElement("ol", "tool-timeline");
      safeTrace.forEach((item) => {
        const entry = createElement("li");
        const toolBlock = createElement("div");
        toolBlock.append(createElement("span", "tool-name", valueAsText(item.tool, "unknown_tool")));

        const status = valueAsText(item.status, "unknown").toLowerCase();
        const statusBadge = createElement("span", "tool-status", status);
        if (["error", "failed", "failure"].includes(status)) {
          statusBadge.classList.add("is-error");
        } else if (["partial", "truncated", "warning"].includes(status)) {
          statusBadge.classList.add("is-partial");
        }
        toolBlock.append(statusBadge);
        entry.append(
          toolBlock,
          createElement("div", "tool-summary", valueAsText(item.summary, "未提供摘要")),
        );
        timeline.append(entry);
      });
      content.append(timeline);
    } else {
      content.append(createElement("p", "tool-summary", "本轮服务未返回工具调用轨迹。"));
    }

    if (safeCitations.length) {
      content.append(createElement("h3", "trace-heading", "源码引用"));
      const list = createElement("ul", "citation-list");
      safeCitations.forEach((citation) => {
        const entry = createElement("li");
        const reference = createElement("code", "", citationLabel(citation));
        reference.title = citationLabel(citation);
        entry.append(
          reference,
          createElement("span", "citation-revision", shortValue(citation.revision, 10)),
        );
        list.append(entry);
      });
      content.append(list);
    }

    details.append(content);
    return details;
  }

  function appendAssistantMessage(data) {
    const article = createElement("article", "message-entry is-assistant");
    const label = createElement("div", "message-label");
    label.append(createElement("span", "", `Agent · ${valueAsText(data.status, "completed")}`));
    if (data.request_id) {
      const requestId = createElement("span", "message-request-id", shortValue(data.request_id, 18));
      requestId.title = valueAsText(data.request_id);
      label.append(requestId);
    }

    const body = createElement("div", "message-body");
    body.append(
      renderAssistantAnswer(data.answer),
      renderEvidenceTrace(data.trace, data.citations),
    );
    article.append(label, body);
    elements.messageLog.append(article);
    scrollMessagesToEnd();
  }

  function appendErrorMessage(message) {
    const article = createElement("article", "message-entry is-error");
    article.append(createElement("div", "message-label", "请求未完成"));
    const body = createElement("div", "message-body");
    body.append(createElement("p", "answer-text", message));
    article.append(body);
    elements.messageLog.append(article);
    scrollMessagesToEnd();
  }

  async function sendQuestion(question) {
    if (!state.sessionId || state.busy) {
      return;
    }

    appendUserMessage(question);
    const pendingMessage = appendPendingMessage();
    setBusy(true, "chat");
    try {
      const data = await requestJson("/api/chat", {
        method: "POST",
        body: {
          session_id: state.sessionId,
          question,
        },
      });
      if (data.session_id && data.session_id !== state.sessionId) {
        state.sessionId = valueAsText(data.session_id);
        updateRepositoryView();
      }
      pendingMessage.remove();
      appendAssistantMessage(data);
    } catch (error) {
      pendingMessage.remove();
      appendErrorMessage(error.message);
      showToast(error.message, "error");
    } finally {
      setBusy(false);
      elements.questionInput.focus();
    }
  }

  async function resetSession() {
    if (!state.sessionId || state.busy) {
      return;
    }

    setBusy(true);
    try {
      const data = await requestJson("/api/session/reset", {
        method: "POST",
        body: { session_id: state.sessionId },
      });
      if (data.ok === false) {
        throw new Error("服务未能清空当前会话。");
      }
      if (data.session_id) {
        state.sessionId = valueAsText(data.session_id);
      }
      updateRepositoryView();
      resetMessageView();
      showToast("当前对话已清空，仓库仍保持连接。", "success");
      window.requestAnimationFrame(() => elements.questionInput.focus());
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      setBusy(false);
    }
  }

  function setBrowserError(message = "") {
    elements.browserError.hidden = !message;
    elements.browserError.textContent = message;
  }

  function renderDirectoryEntries(entries) {
    const directories = (Array.isArray(entries) ? entries : [])
      .filter((entry) => entry && entry.is_dir !== false)
      .sort((left, right) =>
        valueAsText(left.name).localeCompare(valueAsText(right.name), "zh-CN", {
          numeric: true,
          sensitivity: "base",
        }),
      );

    elements.directoryList.replaceChildren();
    if (!directories.length) {
      elements.directoryList.append(
        createElement("div", "directory-empty", "当前目录中没有可进入的子目录。"),
      );
      return;
    }

    directories.forEach((entry) => {
      const button = createElement("button", "directory-entry");
      button.type = "button";
      button.title = valueAsText(entry.path, entry.name);
      button.append(
        createElement("span", "directory-icon"),
        createElement("span", "directory-name", valueAsText(entry.name, pathBasename(entry.path))),
        createElement("span", "directory-arrow", "→"),
      );
      button.addEventListener("click", () => loadDirectory(entry.path));
      elements.directoryList.append(button);
    });
  }

  async function loadDirectory(path) {
    if (!elements.repositoryBrowser.open) {
      return;
    }

    setBrowserError();
    elements.directoryList.setAttribute("aria-busy", "true");
    elements.directoryList.replaceChildren(
      createElement("div", "directory-loading", "正在读取目录…"),
    );
    elements.parentDirectory.disabled = true;
    elements.selectCurrentDirectory.disabled = true;

    try {
      const query = new URLSearchParams();
      if (path) {
        query.set("path", valueAsText(path));
      }
      const data = await requestJson(`/api/fs?${query.toString()}`);
      state.browsePath = valueAsText(data.path, path);
      state.browseParent = data.parent ? valueAsText(data.parent) : null;
      elements.browserPath.textContent = state.browsePath || "—";
      elements.browserPath.title = state.browsePath;
      elements.parentDirectory.disabled = !state.browseParent;
      elements.selectCurrentDirectory.disabled = !state.browsePath;
      renderDirectoryEntries(data.entries);
    } catch (error) {
      setBrowserError(error.message);
      elements.directoryList.replaceChildren(
        createElement("div", "directory-empty", "无法显示此目录。"),
      );
    } finally {
      elements.directoryList.setAttribute("aria-busy", "false");
    }
  }

  function openRepositoryBrowser() {
    setBrowserError();
    if (typeof elements.repositoryBrowser.showModal === "function") {
      elements.repositoryBrowser.showModal();
    } else {
      elements.repositoryBrowser.setAttribute("open", "");
    }
    const initialPath =
      elements.repositoryPath.value.trim() ||
      valueAsText(state.repository?.path) ||
      valueAsText(state.config?.default_browse_path);
    loadDirectory(initialPath);
  }

  elements.repositoryForm.addEventListener("submit", (event) => {
    event.preventDefault();
    selectRepository(elements.repositoryPath.value);
  });

  elements.browseButton.addEventListener("click", openRepositoryBrowser);

  elements.parentDirectory.addEventListener("click", () => {
    if (state.browseParent) {
      loadDirectory(state.browseParent);
    }
  });

  elements.selectCurrentDirectory.addEventListener("click", async () => {
    if (state.browsePath) {
      elements.repositoryPath.value = state.browsePath;
      await selectRepository(state.browsePath);
    }
  });

  elements.resetSession.addEventListener("click", resetSession);

  elements.chatComposer.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = elements.questionInput.value.trim();
    if (!question || state.busy || !state.sessionId) {
      updateComposerState();
      return;
    }
    elements.questionInput.value = "";
    autoResizeQuestion();
    updateComposerState();
    sendQuestion(question);
  });

  elements.questionInput.addEventListener("input", () => {
    autoResizeQuestion();
    updateComposerState();
  });

  elements.questionInput.addEventListener("keydown", (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing &&
      event.keyCode !== 229
    ) {
      event.preventDefault();
      if (!elements.sendButton.disabled) {
        elements.chatComposer.requestSubmit();
      }
    }
  });

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!state.sessionId) {
        showToast("请先载入一个代码仓库。", "error");
        elements.repositoryPath.focus();
        return;
      }
      elements.questionInput.value = valueAsText(button.dataset.prompt);
      autoResizeQuestion();
      updateComposerState();
      elements.questionInput.focus();
    });
  });

  elements.repositoryBrowser.addEventListener("close", () => {
    setBrowserError();
  });

  updateRepositoryView();
  updateComposerState();
  loadConfig();
})();
