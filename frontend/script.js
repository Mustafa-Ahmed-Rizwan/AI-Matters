document.addEventListener("DOMContentLoaded", () => {
   // ----------------- Element Selections -----------------
  const searchInput = document.getElementById("searchInput");
  const searchButton = document.getElementById("searchButton");
  const heroSection = document.getElementById("heroSection");
  const chatFlow = document.getElementById("chatFlow");
  const conversationHistory = document.getElementById("conversationHistory");
  const quickHelpContainer = document.getElementById("quickHelpContainer");
  
  const chatInput = document.getElementById("chatInput");
  const chatSendButton = document.getElementById("chatSendButton");
  const relatedQuestionsSection = document.getElementById(
    "relatedQuestionsSection"
  );
  const relatedQuestionsList = document.getElementById("relatedQuestionsList");

    // ----------------- State Variables -----------------
  let suggestionTimeout;
  let currentSuggestions = [];
  let selectedSuggestionIndex = -1;
  let suggestionsVisible = false;
  let virtualList = null;
  let messageCounter = 0;
  let chatSuggestionsVisible = false;
  let chatSelectedSuggestionIndex = -1;
  let currentChatSuggestions = [];
  let chatVirtualList = null;
  let isAnswerLoading = false;
  let currentAbortController = null;
  
function animateElement(element, animationClass, duration = 500) {
  return new Promise((resolve) => {
    element.classList.add(animationClass);
    setTimeout(() => {
      element.classList.remove(animationClass);
      resolve();
    }, duration);
  });
}

function staggerAnimation(elements, delay = 100) {
  elements.forEach((element, index) => {
    setTimeout(() => {
      element.style.animation = `fadeInUp 0.5s ease-out both`;
    }, index * delay);
  });
}
/**
 * Update hero title with time-based greeting
 * @param {string} lang - Language code ('EN' or 'DE')
 */
function updateHeroTitle() {
  const heroTitle = document.querySelector('.hero-title');
  if (!heroTitle) return;

  const now = new Date();
  const hour = now.getHours();

  let greeting;
  if (hour >= 5 && hour < 12) {
    greeting = 'Good morning! How can we help?';
  } else if (hour >= 12 && hour < 18) {
    greeting = 'Good afternoon! How can we help?';
  } else if (hour >= 18 && hour < 22) {
    greeting = 'Good evening! How can we help?';
  } else {
    greeting = 'Good night! How can we help?';
  }

  heroTitle.textContent = greeting;
}

    // ----------------- Marked.js Configuration -----------------
  marked.setOptions({
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true,
  });
  // ----------------- Quick-help show/hide helpers -----------------
function hideQuickHelp() {
  if (quickHelpContainer) quickHelpContainer.classList.add('hidden');
}
function showQuickHelp() {
  if (quickHelpContainer) quickHelpContainer.classList.remove('hidden');
}

    // ----------------- Cache Management -----------------
    /**
     * IndexedDB Cache Manager for storing suggestions
     */
  class SuggestionCache {
    constructor() {
      this.dbName = "Arena2036SuggestionsDB";
      this.dbVersion = 1;
      this.storeName = "suggestions";
      this.db = null;
      this.memoryCache = new Map();
      this.maxMemorySize = 500;
      this.cacheExpiry = 24 * 60 * 60 * 1000; // 24 hours
      this.init();
    }

    async init() {
      try {
        this.db = await this.openDB();
        console.log("IndexedDB initialized successfully");
      } catch (error) {
        console.warn("IndexedDB failed to initialize:", error);
      }
    }

    openDB() {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open(this.dbName, this.dbVersion);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          if (!db.objectStoreNames.contains(this.storeName)) {
            const store = db.createObjectStore(this.storeName, {
              keyPath: "query",
            });
            store.createIndex("timestamp", "timestamp", { unique: false });
          }
        };
      });
    }

    async get(query, language = "EN") {
      const cacheKey = query.toLowerCase().trim();

      // Check memory cache first
      if (this.memoryCache.has(cacheKey)) {
        const cached = this.memoryCache.get(cacheKey);
        if (Date.now() - cached.timestamp < this.cacheExpiry) {
          return cached.suggestions;
        } else {
          this.memoryCache.delete(cacheKey);
        }
      }

      // Check IndexedDB
      if (!this.db) return null;

      try {
        const transaction = this.db.transaction([this.storeName], "readonly");
        const store = transaction.objectStore(this.storeName);

        return new Promise((resolve) => {
          const request = store.get(cacheKey);

          request.onsuccess = () => {
            const result = request.result;
            if (result && Date.now() - result.timestamp < this.cacheExpiry) {
              // Update memory cache
              this.memoryCache.set(cacheKey, {
                suggestions: result.suggestions,
                timestamp: result.timestamp,
              });
              resolve(result.suggestions);
            } else {
              resolve(null);
            }
          };

          request.onerror = () => resolve(null);
        });
      } catch (error) {
        console.warn("Cache get error:", error);
        return null;
      }
    }

    async set(query, suggestions, language = "EN") {
      const cacheKey = query.toLowerCase().trim();
      const timestamp = Date.now();
      const cacheData = { suggestions, timestamp };

      // Update memory cache
      this.memoryCache.set(cacheKey, cacheData);

      // Limit memory cache size
      if (this.memoryCache.size > this.maxMemorySize) {
        const firstKey = this.memoryCache.keys().next().value;
        this.memoryCache.delete(firstKey);
      }

      // Update IndexedDB
      if (!this.db) return;

      try {
        const transaction = this.db.transaction([this.storeName], "readwrite");
        const store = transaction.objectStore(this.storeName);

        store.put({
          query: cacheKey,
          suggestions,
          timestamp,
        });
      } catch (error) {
        console.warn("Cache set error:", error);
      }
    }

    async clear() {
      this.memoryCache.clear();
      if (!this.db) return;

      try {
        const transaction = this.db.transaction([this.storeName], "readwrite");
        const store = transaction.objectStore(this.storeName);
        store.clear();
      } catch (error) {
        console.warn("Cache clear error:", error);
      }
    }
  }

    // ----------------- Virtual Scrolling Implementation -----------------
    /**
     * Virtual list for efficient rendering of large suggestion lists
     */
  class VirtualSuggestionsList {
    constructor(container, inputElement = null) {
      this.container = container;
      this.inputElement = inputElement; // ADD THIS
      this.itemHeight = 44;
      this.visibleCount = Math.min(8, Math.floor(300 / this.itemHeight));
      this.scrollTop = 0;
      this.data = [];
      this.selectedIndex = -1;
    }

    render(suggestions, selectedIndex = -1) {
      this.data = suggestions;
      this.selectedIndex = selectedIndex;

      if (suggestions.length === 0) {
        this.container.innerHTML = `<div class="no-suggestions">No suggestions found</div>`;
        return;
      }

      if (suggestions.length <= 10) {
        this.renderDirect(suggestions, selectedIndex);
        return;
      }

      const totalHeight = suggestions.length * this.itemHeight;
      const startIndex = Math.floor(this.scrollTop / this.itemHeight);
      const endIndex = Math.min(
        startIndex + this.visibleCount + 2,
        suggestions.length
      );

      const visibleItems = suggestions.slice(startIndex, endIndex);

      this.container.innerHTML = `
                <div class="virtual-container" style="height: ${totalHeight}px; position: relative;">
                    <div class="virtual-content" style="transform: translateY(${
                      startIndex * this.itemHeight
                    }px);">
                        ${visibleItems
                          .map((suggestion, index) => {
                            const actualIndex = startIndex + index;
                            const isSelected = actualIndex === selectedIndex;
                            return `<div class="suggestion-item ${
                              isSelected ? "selected" : ""
                            }" 
                                        data-index="${actualIndex}" 
                                        style="height: ${
                                          this.itemHeight
                                        }px; line-height: ${
                              this.itemHeight
                            }px;">
                                        <span class="suggestion-icon">🔍</span>
                                        <span class="suggestion-text">${this.highlightMatch(
                                          suggestion,
                                          this.inputElement ? this.inputElement.value : "" // CHANGE THIS
                                        )}</span>
                                    </div>`;
                          })
                          .join("")}
                    </div>
                </div>
            `;
    }

    renderDirect(suggestions, selectedIndex = -1) {
      this.container.innerHTML = suggestions
        .map((suggestion, index) => {
          const isSelected = index === selectedIndex;
          return `<div class="suggestion-item ${isSelected ? "selected" : ""}" 
                            data-index="${index}">
                            <span class="suggestion-icon">🔍</span>
                            <span class="suggestion-text">${this.highlightMatch(
                              suggestion,
                              this.inputElement ? this.inputElement.value : "" // CHANGE THIS
                            )}</span>
                        </div>`;
        })
        .join("");
    }

    highlightMatch(text, query) {
      if (!query || query.length < 2) return text;

      const regex = new RegExp(
        `(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`,
        "gi"
      );
      return text.replace(regex, "<mark>$1</mark>");
    }

    handleScroll(scrollTop) {
      if (this.data.length <= 10) return;

      this.scrollTop = scrollTop;
      this.render(this.data, this.selectedIndex);
    }

    updateSelection(newIndex) {
      this.selectedIndex = newIndex;
      this.render(this.data, newIndex);

      if (this.data.length > 10) {
        const itemTop = newIndex * this.itemHeight;
        const itemBottom = itemTop + this.itemHeight;
        const containerTop = this.scrollTop;
        const containerBottom =
          containerTop + this.visibleCount * this.itemHeight;

        if (itemTop < containerTop) {
          this.container.scrollTop = itemTop;
        } else if (itemBottom > containerBottom) {
          this.container.scrollTop =
            itemBottom - this.visibleCount * this.itemHeight;
        }
      }
    }
  }

    // ----------------- Initialize Cache -----------------
  const cache = new SuggestionCache();

    // ----------------- Quick Help Suggestions -----------------
    /**
     * Load quick help suggestions based on current language
     */ 
  loadQuickHelpSuggestions();

  // Main functions
async function loadQuickHelpSuggestions() {
  try {
    const cachedSuggestions = await cache.get('__quick_help__');
    if (cachedSuggestions && cachedSuggestions.length > 0) {
      renderQuickHelp(cachedSuggestions.slice(0, 5));
      return;
    }

    const response = await fetch(`http://localhost:8000/suggestions?limit=5`);
    if (response.ok) {
      const data = await response.json();
      await cache.set('__quick_help__', data.suggestions);
      renderQuickHelp(data.suggestions);
    } else {
      throw new Error('API request failed');
    }
  } catch (error) {
    console.error("Error loading suggestions:", error);
    const fallback = [
    "What is ARENA2036 about?",
    "How can I learn about the ARENA2036 project?",
    "What does ARENA2036 offer for me?",
    "How do I use ARENA2036 Projects?",
    "How do I customize my ARENA2036 profile?"
    ];
    renderQuickHelp(fallback);
  }
}


function renderQuickHelp(suggestions) {
  const quickHelpLoadingEl = quickHelpContainer.querySelector("#quickHelpLoading");
  if (quickHelpLoadingEl) {
    // Fade out loading
    quickHelpLoadingEl.style.transition = 'opacity 0.3s ease';
    quickHelpLoadingEl.style.opacity = '0';
    setTimeout(() => {
      quickHelpLoadingEl.style.display = "none";
    }, 300);
  }

  quickHelpContainer.innerHTML = "";
  
  suggestions.forEach((suggestion, index) => {
    const btn = document.createElement("button");
    btn.className = "help-tag";
    btn.dataset.question = suggestion;
    btn.textContent = suggestion;
    btn.style.opacity = '0';
    btn.style.transform = 'translateY(20px)';
    quickHelpContainer.appendChild(btn);
    
    // Stagger animation
    setTimeout(() => {
      btn.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
      btn.style.opacity = '1';
      btn.style.transform = 'translateY(0)';
    }, index * 100);
  });

  quickHelpContainer.querySelectorAll(".help-tag").forEach((tag) => {
    tag.addEventListener("click", () => {
      startChatFlow(tag.dataset.question);
    });
  });
}


    // ----------------- Autocomplete Functionality -----------------
    /**
     * Unified autocomplete handler for both search and chat inputs
     * @param {Object} params - Configuration object
     * @param {string} params.query - Search query
     * @param {string} params.inputType - Input type ('search' or 'chat')
     */
  async function loadUnifiedAutocomplete({
    query,
    inputType, // 'search' or 'chat'
  }) {
    if (query.length < 1) {
      if (inputType === "search") hideSuggestions();
      else hideChatSuggestions();
      return;
    }

    try {
      let suggestions = null;
      if (inputType === "search") {
        const cachedSuggestions = await cache.get(query);
        if (cachedSuggestions) {
          suggestions = cachedSuggestions;
        } else {
          const response = await fetch(
`http://localhost:8000/suggestions?q=${encodeURIComponent(query)}&limit=20`
          );
          if (response.ok) {
            const data = await response.json();
            await cache.set(query, data.suggestions);
            suggestions = data.suggestions;
          }
        }
        showUnifiedAutocompleteSuggestions(suggestions || [], "search");
      } else {
        // chat input does not use cache
        const response = await fetch(
          `http://localhost:8000/suggestions?q=${encodeURIComponent(
            query
          )}&limit=20`
        );
        if (response.ok) {
          const data = await response.json();
          suggestions = data.suggestions;
        }
        showUnifiedAutocompleteSuggestions(suggestions || [], "chat");
      }
    } catch (error) {
      console.error("Error loading autocomplete suggestions:", error);
      if (inputType === "search") hideSuggestions();
      else hideChatSuggestions();
    }
  }

    /**
     * Show autocomplete suggestions
     * @param {Array} suggestions - List of suggestions
     * @param {string} inputType - Input type ('search' or 'chat')
     */
function showUnifiedAutocompleteSuggestions(suggestions, inputType) {
  if (inputType === "search") {
    if (suggestions.length === 0) {
      hideSuggestions();
      return;
    }
    currentSuggestions = suggestions;
    selectedSuggestionIndex = -1;
    let suggestionsDropdown = document.getElementById("suggestionsDropdown");
    if (!suggestionsDropdown) {
      suggestionsDropdown = document.createElement("div");
      suggestionsDropdown.id = "suggestionsDropdown";
      suggestionsDropdown.className = "suggestions-dropdown";
      suggestionsDropdown.style.opacity = '0';
      suggestionsDropdown.style.transform = 'translateY(-10px)';
      document.querySelector(".search-wrapper").appendChild(suggestionsDropdown);
      virtualList = new VirtualSuggestionsList(suggestionsDropdown, searchInput);
        suggestionsDropdown.addEventListener("scroll", (e) => {
          virtualList.handleScroll(e.target.scrollTop);
        });
        suggestionsDropdown.addEventListener("click", handleSuggestionClick);
      }
      hideQuickHelp();
      virtualList.render(suggestions);
      suggestionsDropdown.style.display = "block";
      setTimeout(() => {
      suggestionsDropdown.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
      suggestionsDropdown.style.opacity = '1';
      suggestionsDropdown.style.transform = 'translateY(0)';
    }, 10);
    
      suggestionsVisible = true;
    } else {
      if (suggestions.length === 0 || isAnswerLoading) {
        hideChatSuggestions();
        return;
      }
      currentChatSuggestions = suggestions;
      chatSelectedSuggestionIndex = -1;
      let chatSuggestionsDropdown = document.getElementById(
        "chatSuggestionsDropdown"
      );
      if (!chatSuggestionsDropdown) {
        chatSuggestionsDropdown = document.createElement("div");
        chatSuggestionsDropdown.id = "chatSuggestionsDropdown";
        chatSuggestionsDropdown.className = "chat-suggestions-dropdown";
        document
          .querySelector(".chat-input-container")
          .appendChild(chatSuggestionsDropdown);
        chatVirtualList = new VirtualSuggestionsList(chatSuggestionsDropdown, chatInput);
        chatSuggestionsDropdown.addEventListener("scroll", (e) => {
          chatVirtualList.handleScroll(e.target.scrollTop);
        });
        chatSuggestionsDropdown.addEventListener(
          "click",
          handleChatSuggestionClick
        );
      }
      chatVirtualList.render(suggestions);
      chatSuggestionsDropdown.style.display = "block";
      chatSuggestionsVisible = true;
    }
  }

    // ----------------- Suggestion Visibility Management -----------------
  function hideSuggestions() {
    const suggestionsDropdown = document.getElementById("suggestionsDropdown");
    if (suggestionsDropdown) {
      suggestionsDropdown.style.display = "none";
    }
    suggestionsVisible = false;
    selectedSuggestionIndex = -1;

    // If user hasn't typed anything, bring quick-help back
    if (searchInput && searchInput.value.trim().length === 0) {
      showQuickHelp();
    } else {
      // if there's still text, keep quick help hidden (the dropdown would normally handle visibility)
      hideQuickHelp();
    }
  }


  function hideChatSuggestions() {
    const chatSuggestionsDropdown = document.getElementById(
      "chatSuggestionsDropdown"
    );
    if (chatSuggestionsDropdown) {
      chatSuggestionsDropdown.style.display = "none";
    }
    chatSuggestionsVisible = false;
    chatSelectedSuggestionIndex = -1;
  }

    // ----------------- Suggestion Interaction Handlers -----------------
  function handleSuggestionClick(e) {
    const item = e.target.closest(".suggestion-item");
    if (item) {
      const index = parseInt(item.dataset.index, 10);
      const suggestion = currentSuggestions[index];
      searchInput.value = suggestion;
      hideSuggestions();
      startChatFlow(suggestion);
    }
  }

  function handleChatSuggestionClick(e) {
    const item = e.target.closest(".suggestion-item");
    if (item) {
      const index = parseInt(item.dataset.index, 10);
      const suggestion = currentChatSuggestions[index];
      chatInput.value = suggestion;
      hideChatSuggestions();
      addChatMessage(suggestion);
      chatInput.value = "";
    }
  }

  function handleSuggestionKeyboard(e) {
    if (!suggestionsVisible || currentSuggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedSuggestionIndex = Math.min(
        selectedSuggestionIndex + 1,
        currentSuggestions.length - 1
      );
      virtualList.updateSelection(selectedSuggestionIndex);
      searchInput.value = currentSuggestions[selectedSuggestionIndex];
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, -1);
      if (selectedSuggestionIndex >= 0) {
        virtualList.updateSelection(selectedSuggestionIndex);
        searchInput.value = currentSuggestions[selectedSuggestionIndex];
      } else {
        virtualList.updateSelection(-1);
        searchInput.value = searchInput.dataset.originalValue || "";
      }
    } else if (e.key === "Enter" && selectedSuggestionIndex >= 0) {
      e.preventDefault();
      const selectedSuggestion = currentSuggestions[selectedSuggestionIndex];
      searchInput.value = selectedSuggestion;
      hideSuggestions();
      startChatFlow(selectedSuggestion);
    } else if (e.key === "Escape") {
      hideSuggestions();
      searchInput.value = searchInput.dataset.originalValue || "";
    }
  }

  function handleChatSuggestionKeyboard(e) {
    if (!chatSuggestionsVisible || currentChatSuggestions.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      chatSelectedSuggestionIndex = Math.min(
        chatSelectedSuggestionIndex + 1,
        currentChatSuggestions.length - 1
      );
      chatVirtualList.updateSelection(chatSelectedSuggestionIndex);
      chatInput.value = currentChatSuggestions[chatSelectedSuggestionIndex];
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      chatSelectedSuggestionIndex = Math.max(
        chatSelectedSuggestionIndex - 1,
        -1
      );
      if (chatSelectedSuggestionIndex >= 0) {
        chatVirtualList.updateSelection(chatSelectedSuggestionIndex);
        chatInput.value = currentChatSuggestions[chatSelectedSuggestionIndex];
      } else {
        chatVirtualList.updateSelection(-1);
        chatInput.value = chatInput.dataset.originalValue || "";
      }
    } else if (e.key === "Enter" && chatSelectedSuggestionIndex >= 0) {
      e.preventDefault();
      const selectedSuggestion =
        currentChatSuggestions[chatSelectedSuggestionIndex];
      chatInput.value = selectedSuggestion;
      hideChatSuggestions();
      addChatMessage(selectedSuggestion);
      chatInput.value = "";
    } else if (e.key === "Escape") {
      hideChatSuggestions();
      chatInput.value = chatInput.dataset.originalValue || "";
    }
  }

    // ----------------- Chat Flow Management -----------------
    /**
     * Start chat flow with a question
     * @param {string} question - User's question
     */
function startChatFlow(question) {
  // Animate hero section out
  heroSection.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
  heroSection.style.transform = 'translateY(-20px)';
  heroSection.style.opacity = '0';
  
  setTimeout(() => {
    heroSection.style.display = "none";
    chatFlow.style.display = "block";
    chatFlow.style.opacity = '0';
    chatFlow.style.transform = 'translateY(20px)';
    
    // Animate chat flow in
    setTimeout(() => {
      chatFlow.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
      chatFlow.style.opacity = '1';
      chatFlow.style.transform = 'translateY(0)';
    }, 50);
    
    addChatMessage(question);

    setTimeout(() => {
      chatInput.focus();
      scrollToBottom();
    }, 400);
  }, 200);
}
function addChatMessage(question) {
    if (isAnswerLoading) return;

    messageCounter++;
    const messageId = `message-${messageCounter}`;

    isAnswerLoading = true;
    updateInputStates();

    const messageElement = document.createElement("div");
    messageElement.className = "chat-message";
    messageElement.id = messageId;

    // Detect image requests (now includes German keywords)
    const isImageRequest =
    /\b(?:image|picture|photo|illustration|draw|render)\b/i.test(question);   

    // Set messages based on language and request type
    const headerIcon = isImageRequest ? "🖼️" : "✨";
    const headerLabel = isImageRequest ? 'Creating your image' : 'Answer';
    const loadingMessage = isImageRequest
    ? 'Hang tight, your image is being crafted...'
    : 'Finding the best answer for you...';


    messageElement.innerHTML = `
        <div class="question-display">${question}</div>
        <div class="answer-section ${isImageRequest ? "image-mode" : ""}">
            <div class="answer-header">
                <span class="answer-icon">${headerIcon}</span>
                <span class="answer-label">${headerLabel}</span>
                <button aria-label="Stop generation" class="stop-generation-btn" onclick="stopGeneration('${messageId}')">
                    <svg viewBox="0 0 24 24" width="16" height="16">
                        <rect x="8" y="8" width="8" height="8" fill="currentColor" rx="2"/>
                    </svg>
                </button>
            </div>
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p class="loading-text">${loadingMessage}</p>
                <div class="loading-shimmer"></div>
                <div class="loading-shimmer"></div>
                <div class="loading-shimmer" style="width: 70%;"></div>
            </div>
        </div>
    `;

    conversationHistory.appendChild(messageElement);
    fetchAnswerForMessage(question, messageId);
    setTimeout(() => {
        scrollToBottom();
    }, 100);
}

    /**
     * Fetch answer for a chat message
     * @param {string} question - User's question
     * @param {string} messageId - DOM element ID of the message
     */
const fetchAnswerForMessage = async (question, messageId) => {
  const messageElement = document.getElementById(messageId);
  if (!messageElement) {
    console.warn('Message element removed before response', messageId);
    return;
  }
  const answerSection = messageElement.querySelector(".answer-section");

  // create per-request abort controller and keep it global (re-used by stopGeneration)
  currentAbortController = new AbortController();
  const signal = currentAbortController.signal;

  try {
    const response = await fetch(
      `http://localhost:8000/query?question=${encodeURIComponent(question)}`,
      { signal }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // if it is an image, renderAnswerInMessage will handle it (it returns a Promise)
    await renderAnswerInMessage(
      answerSection,
      data.answer,
      data.sources,
      question,
      data.is_image,
      { signal, typingSpeed: 12, enableClickToSkip: true }
    );

  } catch (error) {
    if (error.name === "AbortError") {
      const abortMd = "## Answer Stopped\n\nGeneration was stopped by user.";


      // render the abort message (await it so controller remains until render finishes)
      await renderAnswerInMessage(answerSection, abortMd, [], question, false, { signal: null, typingSpeed: 0 });
    } else {
      console.error("Error:", error);
      let errorMessage = "I'm having trouble connecting right now. Please try again in a moment.";
      if (error.message.includes("Failed to fetch")) {
        errorMessage = "Unable to connect to the server. Please check your connection and try again.";
     } else if (error.message.includes("500")) {
        errorMessage = "The server encountered an internal error. Please try again later.";
        }


      await renderAnswerInMessage(
        answerSection,
        `## Error\n\n${errorMessage}`,
        [],
        question,
        false,
        { signal: null, typingSpeed: 0 }
      );
    }
  } finally {
    // mark loading false only after rendering completes
    isAnswerLoading = false;
    currentAbortController = null;
    updateInputStates();
  }
};


/**
 * Render answer in message container (async because of typewriter)
 * @param {HTMLElement} answerSection - DOM element to render into
 * @param {string} markdownText - Answer in markdown format
 * @param {Array} sources - List of source objects
 * @param {string} question - Original question
 * @param {boolean} isImage - Whether answer is an image
 * @param {Object} opts - { signal: AbortSignal|null, typingSpeed: ms per char/token (optional), enableClickToSkip: bool (optional) }
 */
async function renderAnswerInMessage(answerSection, markdownText, sources, question, isImage = false, opts = {}) {
  const ENABLE_CLICK_TO_SKIP = opts.enableClickToSkip ?? true;

  // tuning
  const CHAR_THRESHOLD = 3000; // if finalHtml length > this, use word mode
  const CHAR_SPEED_MS = 8;     // ms per character (char mode)
  const WORD_SPEED_MS = 30;    // ms per token (word mode) - tune to taste

  // Clear any existing content and build answer header (synchronous)
  answerSection.innerHTML = '';

  const header = document.createElement('div');
  header.className = 'answer-header';
  header.innerHTML = `
  <span class="answer-icon">✨</span>
  <span class="answer-label">Answer</span>
`;

  answerSection.appendChild(header);

  if (isImage) {
    // Image handling (same logic as before, but allow external URLs or local paths)
    const imgMatch = markdownText && markdownText.match(/!\[.*\]\((.*?)\)/);
    const altMatch = markdownText && markdownText.match(/!\[(.*?)\]/);
    const rawUrl = imgMatch ? imgMatch[1] : null;
    const altText = altMatch ? altMatch[1] : 'Generated Image';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'answer-content';
    answerSection.appendChild(contentDiv);

    if (!rawUrl) {
      contentDiv.innerHTML = `<p>No image URL found.</p>`;
      scrollToBottom();
      return;
    }

    // determine final image src: external URL/data URI used directly, otherwise call serve-image
    let finalImgSrc;
    if (/^data:/.test(rawUrl) || /^https?:\/\//i.test(rawUrl)) {
      finalImgSrc = rawUrl;
      contentDiv.innerHTML = `<img src="${finalImgSrc}" alt="${altText}" style="max-width:100%; border-radius:8px; margin-bottom:12px;"><p>${altText}</p>`;
      scrollToBottom();
      return;
    } else {
      // treat as backend path
      const proxyUrl = `http://localhost:8000/serve-image?path=${encodeURIComponent(rawUrl)}`;
      // show loading UI
      contentDiv.innerHTML = `
        <div class="loading-state">
          <div class="loading-spinner"></div>
          <p class="loading-text">Loading image...</p>
        </div>
      `;
      scrollToBottom();
      try {
        const resp = await fetch(proxyUrl);
        if (!resp.ok) throw new Error('Image fetch failed');
        const blob = await resp.blob();
        const reader = new FileReader();
        const htmlAfter = await new Promise((resolve, reject) => {
          reader.onloadend = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        contentDiv.innerHTML = `<img src="${htmlAfter}" alt="${altText}" style="max-width:100%; border-radius:8px; margin-bottom:12px;"><p>${altText}</p>`;
        scrollToBottom();
        return;
      } catch (err) {
        console.error("Image fetch error:", err);
        contentDiv.innerHTML = `<p>Error loading image.</p>`;
        scrollToBottom();
        return;
      }
    }
  }

  // Non-image: convert markdown to HTML
  const finalHtml = marked.parse(markdownText || '');

  // decide mode based on finalHtml length (characters)
  const mode = (finalHtml.length > CHAR_THRESHOLD) ? 'word' : 'char';

  // select speed: prioritize user-provided opts.typingSpeed if present
  let speed;
  if (typeof opts.typingSpeed === 'number') {
    speed = opts.typingSpeed;
  } else {
    speed = (mode === 'char') ? CHAR_SPEED_MS : WORD_SPEED_MS;
  }

  // content container
  const contentEl = document.createElement('div');
  contentEl.className = 'answer-content';
  contentEl.setAttribute('aria-live', 'polite');
  answerSection.appendChild(contentEl);

  // show related-questions loading UI (same UX)
  relatedQuestionsSection.style.display = "block";
  relatedQuestionsList.innerHTML = `
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p class="loading-text">Loading related questions...</p>
    </div>
  `;

  try {
    // call unified typewriter with selected mode and speed
    await typewriteHtml(contentEl, finalHtml, { speed, signal: opts.signal ?? null, enableClickToSkip: ENABLE_CLICK_TO_SKIP, mode });

    // append sources if present
    if (sources && sources.length > 0) {
      const sourcesSection = document.createElement('div');
      sourcesSection.className = 'sources-section';
      sourcesSection.innerHTML = `
        <h4 class="sources-title">Sources</h4>
        <div class="sources-list">
          ${sources.map(s => `
            <a href="${s.url}" target="_blank" rel="noopener noreferrer" class="source-item">
              <span class="source-icon">🔗</span>
              <span class="source-title">${s.title || 'Source'}</span>
            </a>
          `).join('')}
        </div>
      `;
      answerSection.appendChild(sourcesSection);
    }

    // feedback UI
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'answer-feedback';
    feedbackDiv.innerHTML = `
      <span class="feedback-text">Was this answer helpful?</span>
      <div class="feedback-buttons">
        <button class="feedback-btn feedback-yes" aria-label="Helpful">👍</button>
        <button class="feedback-btn feedback-no" aria-label="Not helpful">👎</button>
      </div>
    `;
    answerSection.appendChild(feedbackDiv);

    // highlight code blocks after final HTML inserted
    answerSection.querySelectorAll('pre code').forEach((block) => {
      try { hljs.highlightElement(block); } catch (e) { /* ignore */ }
    });

    // wire feedback buttons
    feedbackDiv.querySelectorAll('.feedback-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.parentNode.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // load related questions asynchronously
    loadRelatedQuestions(question);

    scrollToBottom();
  } catch (err) {
    if (err && (err.name === 'AbortError' || err.message === 'aborted')) {
      // if aborted, reveal final content so user can read it
      const abortNotice = document.createElement('div');
      abortNotice.className = 'answer-content';
      abortNotice.innerHTML = `<blockquote>Output was stopped.</blockquote>`;
      answerSection.appendChild(abortNotice);
      const fullReveal = document.createElement('div');
      fullReveal.className = 'answer-content';
      fullReveal.innerHTML = finalHtml;
      answerSection.appendChild(fullReveal);

      answerSection.querySelectorAll('pre code').forEach((block) => {
        try { hljs.highlightElement(block); } catch (e) { /* ignore */ }
      });
      scrollToBottom();
    } else {
      console.error('Error during typewriting:', err);
      const errDiv = document.createElement('div');
      errDiv.className = 'answer-content';
      errDiv.innerHTML = `<p>Error rendering the answer.</p>`;
      answerSection.appendChild(errDiv);
      scrollToBottom();
    }
  }
}


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Typewrite HTML string into a container while preserving tags.
 * Supports two modes:
 *   - 'char' : reveals characters one-by-one (original behaviour)
 *   - 'word' : reveals tokens (words + whitespace) one token at a time (much faster for long text)
 *
 * @param {HTMLElement} container
 * @param {string} htmlString
 * @param {Object} opts - { speed: ms per char/token, signal: AbortSignal|null, enableClickToSkip: bool, mode: 'char'|'word' }
 */
async function typewriteHtml(container, htmlString, opts = {}) {
  const {
    speed = 12,
    signal = null,
    enableClickToSkip = true,
    mode = 'char' // 'char' or 'word'
  } = opts || {};

  // Build DOM nodes with empty text nodes while preserving element structure
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlString, 'text/html');
  const sourceBody = doc.body;

  const textNodes = []; // { tokens: Array<string>, targetNode: Text }

  function cloneStructureWithEmptyText(srcNode, destParent) {
    for (const child of Array.from(srcNode.childNodes)) {
      if (child.nodeType === Node.TEXT_NODE) {
        const original = child.nodeValue || '';

        // tokenize depending on mode
        if (mode === 'char') {
          // single token string; we'll iterate chars later
          const tn = document.createTextNode('');
          destParent.appendChild(tn);
          textNodes.push({ tokens: [original], mode: 'char', targetNode: tn });
        } else {
          // mode === 'word': split into tokens that preserve whitespace
          // regex splits into runs of non-whitespace or runs of whitespace
          const tokens = original.match(/\s+|\S+/g) || [];
          const tn = document.createTextNode('');
          destParent.appendChild(tn);
          textNodes.push({ tokens, mode: 'word', targetNode: tn });
        }
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        const el = document.createElement(child.tagName.toLowerCase());
        for (const attr of Array.from(child.attributes || [])) {
          el.setAttribute(attr.name, attr.value);
        }
        destParent.appendChild(el);
        cloneStructureWithEmptyText(child, el);
      } else {
        // ignore other nodes (comments, etc.)
      }
    }
  }

  container.innerHTML = '';
  const fragment = document.createDocumentFragment();
  cloneStructureWithEmptyText(sourceBody, fragment);
  container.appendChild(fragment);

  // click-to-skip logic
  let skipRequested = false;
  function onClickSkip() {
    skipRequested = true;
    container.removeEventListener('click', onClickSkip);
  }
  if (enableClickToSkip) {
    container.addEventListener('click', onClickSkip);
  }

  // main reveal loop
  for (const item of textNodes) {
    if (signal && signal.aborted) {
      throw new DOMException('aborted', 'AbortError');
    }
    if (skipRequested) {
      container.innerHTML = htmlString;
      return;
    }

    if (item.mode === 'char') {
      const text = item.tokens[0];
      // reveal by character
      for (let i = 0; i < text.length; i++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        if (skipRequested) {
          container.innerHTML = htmlString;
          return;
        }
        item.targetNode.data += text[i];
        if (speed > 0) await sleep(speed);
      }
    } else {
      // word mode: tokens already include whitespace tokens; reveal token-by-token
      for (let t = 0; t < item.tokens.length; t++) {
        if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');
        if (skipRequested) {
          container.innerHTML = htmlString;
          return;
        }
        item.targetNode.data += item.tokens[t]; // append whole token (word or whitespace)
        if (speed > 0) await sleep(speed);
      }
    }
  }

  // final safety: ensure container contains final HTML
  container.innerHTML = htmlString;
}


    // ----------------- Related Questions -----------------
    /**
     * Load questions related to the current one
     * @param {string} currentQuestion - The current question
     */
async function loadRelatedQuestions(currentQuestion) {
    try {
        const response = await fetch(
            `http://localhost:8000/related-questions?question=${encodeURIComponent(currentQuestion)}`
        );
        
        if (response.ok) {
            const data = await response.json();
            renderRelatedQuestions(data.related_questions);
        } else {
            // Language-specific fallback questions
            const fallback = [
            "How do I manage Arena2036 notifications?",
            "What are the Arena2036 collaboration features?",
            "How do I integrate third-party tools with Arena2036?",
            "How do I export data from Arena2036?"
            ];

            renderRelatedQuestions(fallback);
        }
    } catch (error) {
        console.error("Error loading related questions:", error);
        // Same fallback as above
        const fallback =   [
            "How do I manage Arena2036 notifications?",
            "What are the Arena2036 collaboration features?",
            "How do I integrate third-party tools with Arena2036?",
            "How do I export data from Arena2036?"
        ];
        renderRelatedQuestions(fallback);
    }
}

function renderRelatedQuestions(questions) {
  relatedQuestionsList.innerHTML = ""; // Clear previous ones
  questions.forEach((q) => {
    const btn = document.createElement('button');
    btn.className = 'related-question';
    btn.dataset.question = q;
    btn.textContent = q;
    relatedQuestionsList.appendChild(btn);
  });

  relatedQuestionsSection.style.display = "block";

  relatedQuestionsList
    .querySelectorAll(".related-question")
    .forEach((questionBtn) => {
      questionBtn.addEventListener("click", () => {
        addChatMessage(questionBtn.dataset.question);
        chatInput.value = "";
      });
    });
}


    // ----------------- Utility Functions -----------------
  function scrollToBottom() {
    setTimeout(() => {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }, 50);
  }

  function stopGeneration(messageId) {
    if (currentAbortController) {
      currentAbortController.abort();
    }

    // Hide stop button
    const stopBtn = document.querySelector(
      `#${messageId} .stop-generation-btn`
    );
    if (stopBtn) {
      stopBtn.style.display = "none";
    }
  }

function updateInputStates() {
  const searchInputContainer = document.querySelector('.search-wrapper');
  const chatInputContainer = document.querySelector('.chat-input-container');

  if (isAnswerLoading) {
    const waitText = "Please wait for the current answer to complete...";

    searchInput.placeholder = waitText;
    chatInput.placeholder = waitText;

    searchInput.disabled = true;
    chatInput.disabled = true;
    searchButton.disabled = true;
    chatSendButton.disabled = true;

    searchInputContainer.classList.add('disabled');
    chatInputContainer.classList.add('disabled');

    hideChatSuggestions();
  } else {
    // Restore placeholders from data-placeholder-en if present, otherwise default text
    searchInput.placeholder = searchInput.getAttribute('data-placeholder-en') || 'Search for answers...';
    chatInput.placeholder = "Ask a follow-up question...";

    searchInput.disabled = false;
    chatInput.disabled = false;
    searchButton.disabled = false;
    chatSendButton.disabled = false;

    searchInputContainer.classList.remove('disabled');
    chatInputContainer.classList.remove('disabled');
  }
}


    // ----------------- Global Functions -----------------
  window.handleFeedback = function (button, isPositive) {
    const feedbackButtons = button.parentNode;
    feedbackButtons
      .querySelectorAll(".feedback-btn")
      .forEach((btn) => btn.classList.remove("active"));
    button.classList.add("active");
  };

  // Make stopGeneration function global
  window.stopGeneration = stopGeneration;

    // ----------------- Event Listeners -----------------
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.trim();
    searchInput.dataset.originalValue = query;
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
    }
    suggestionTimeout = setTimeout(() => {
      loadUnifiedAutocomplete({ query, inputType: "search" });
    }, 150);
  });

  searchInput.addEventListener("keydown", handleSuggestionKeyboard);

  searchInput.addEventListener("focus", () => {
    // hide default quick-help immediately when user focuses the search box
    hideQuickHelp();

    const query = searchInput.value.trim();
    if (query.length >= 1) {
      loadUnifiedAutocomplete({ query, inputType: "search" });
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-wrapper")) {
      hideSuggestions();
    }
    if (!e.target.closest(".chat-input-container")) {
      hideChatSuggestions();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (isAnswerLoading && currentAbortController) {
        stopGeneration(`message-${messageCounter}`);
      }
      hideSuggestions();
      hideChatSuggestions();
    }
  });

  searchButton.addEventListener("click", () => {
    const question = searchInput.value.trim();
    if (question) {
      hideSuggestions();
      startChatFlow(question);
    }
  });

  searchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !suggestionsVisible) {
      const question = searchInput.value.trim();
      if (question) {
        hideSuggestions();
        startChatFlow(question);
      }
    }
  });

  chatInput.addEventListener("input", (e) => {
    if (isAnswerLoading) return;
    const query = e.target.value.trim();
    chatInput.dataset.originalValue = query;
    if (suggestionTimeout) {
      clearTimeout(suggestionTimeout);
    }
    suggestionTimeout = setTimeout(() => {
      loadUnifiedAutocomplete({ query, inputType: "chat" });
    }, 150);
  });

  chatInput.addEventListener("keydown", (e) => {
    if (isAnswerLoading && e.key !== "Escape") {
      e.preventDefault();
      return;
    }
    handleChatSuggestionKeyboard(e);
  });

  chatInput.addEventListener("focus", () => {
    if (isAnswerLoading) return;
    const query = chatInput.value.trim();
    if (query.length >= 1) {
      loadUnifiedAutocomplete({ query, inputType: "chat" });
    }
  });

  chatSendButton.addEventListener("click", () => {
    if (isAnswerLoading) return;

    const question = chatInput.value.trim();
    if (question) {
      hideChatSuggestions();
      addChatMessage(question);
      chatInput.value = "";
    }
  });

  chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !chatSuggestionsVisible && !isAnswerLoading) {
      const question = chatInput.value.trim();
      if (question) {
        hideChatSuggestions();
        addChatMessage(question);
        chatInput.value = "";
      }
    }
  });
    // ----------------- Language Switching -----------------


    // ----------------- Initialization -----------------
    // Focus search input on page load
  // Initialize hero title with time-based greeting
  updateHeroTitle();
  
});