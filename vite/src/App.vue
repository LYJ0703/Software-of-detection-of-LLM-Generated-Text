<template>
  <div class="app-container">
    <header class="header">
      <img src="./AI.svg" alt="AI Text Detector Logo" class="logo" @click="navigateToHomePage" />
      <h1 class="gradient-text" @click="navigateToHomePage">Text Detector</h1>
    </header>
    <main>
      <div class="content-wrapper">
        <div class="left-container">
          <div class="selection">
            <button @click="selectOption('text')" :class="{ active: selectedOption === 'text' }">
              <img src="././assets/text-icon.svg" alt="Text Icon" class="icon"/> 检测文本
            </button>
            <button @click="selectOption('file')" :class="{ active: selectedOption === 'file' }">
              <img src="././assets/upload-icon.svg" alt="Upload Icon" class="icon"/> 上传文件
            </button>
          </div>
          <div v-if="selectedOption === 'text'" class="content-container">
            <div class="textarea-container">
              <textarea v-model="textInput" placeholder="在此输入要分析的文本" rows="15"></textarea>
              <div class="word-count">{{ textInput.length }} / 10000</div>
            </div>
            <div class="button-container">
              <button @click="clearContent">清除</button>
              <button @click="checkContent">检测</button>
              <button @click="quickCheckContent">快速检测</button>
            </div>
          </div>
          <div v-if="selectedOption === 'file'" class="content-container upload-container">
            <div v-if="!fileSelected" class="file-upload-container">
              <input type="file" @change="handleFileUpload" ref="fileInput" class="file-input"/>
              <div class="upload-text">点击上传文件</div>
              <div class="icon-container">
                <img src="./assets/word.svg" alt="Word Icon" class="word-icon"/>
                <img src="./assets/pdf.svg" alt="PDF Icon" class="pdf-icon"/>
              </div>
            </div>
            <div v-else class="file-display-container">
              <div class="file-info">
                <p>{{ file.name }}</p>
                <button @click="removeFile" class="remove-button">
                  <svg class="remove-icon" viewBox="0 0 24 24">
                    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path>
                  </svg>
                </button>
              </div>
              <div class="button-container">
                <button type="submit" class="upload-button" @click="uploadFile">检测</button>
                <button type="submit" class="upload-button" @click="quickUploadFile">快速检测</button>
              </div>
            </div>
          </div>
        </div>
        <div class="right-container">
          <h2>检测结果</h2>
          <div class="result-container" v-if="result !== null">
            <div class="circle-progress">
              <svg viewBox="0 0 36 36" class="circular-chart">
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#b5e7a0;stop-opacity:1" />
                    <stop offset="50%" style="stop-color:#f3e5ab;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#f5b895;stop-opacity:1" />
                  </linearGradient>
                </defs>
                <path class="circle-bg"
                      d="M18 2.0845
                         a 15.9155 15.9155 0 0 1 0 31.831
                         a 15.9155 15.9155 0 0 1 0 -31.831"/>
                <path class="circle"
                      :stroke-dasharray="percentage + ', 100'"
                      d="M18 2.0845
                         a 15.9155 15.9155 0 0 1 0 31.831
                         a 15.9155 15.9155 0 0 1 0 -31.831"
                      stroke="url(#gradient)"/>
                <text x="18" y="15.35" class="percentage">{{ percentage }}%</text>
                <text x="18" y="25.35" class="ai-generated-text">AI Generated</text>
              </svg>
            </div>

            <div class="separator"></div>

            <div class="horizontal-progress">
              <div class="progress-bar">
                <div class="filler left" :style="{ width: (100 - percentage) + '%' }"></div>
                <div class="filler right" :style="{ width: percentage + '%' }"></div>
              </div>
              <div class="percentage-label">
                <div class="label-group">
                  <span class="dot human-dot"></span><span class="label">{{ humanPercentage }}%</span>
                </div>
                <div class="label-group">
                  <span class="dot ai-dot"></span><span class="label">{{ percentage }}%</span>
                </div>
              </div>
            </div>

            <div v-if="result === null" class="keywords">
              <span class="keyword-title">关键字：</span>
              <span class="keyword green">100% 人类</span>
              <span class="keyword yellow">50% 人类</span>
              <span class="keyword red">0% 人类</span>
            </div>
            <div v-else class="result-text">
              LLM Text Detector显示文本有 {{ percentage }}% 的概率是AI生成的。
            </div>
          </div>
          <div class="result-container" v-else>
            <div class="circle-progress">
              <svg viewBox="0 0 36 36" class="circular-chart">
                <defs>
                  <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#b5e7a0;stop-opacity:1" />
                    <stop offset="50%" style="stop-color:#f3e5ab;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#f5b895;stop-opacity:1" />
                  </linearGradient>
                </defs>
                <path class="circle-bg"
                      d="M18 2.0845
                         a 15.9155 15.9155 0 0 1 0 31.831
                         a 15.9155 15.9155 0 0 1 0 -31.831"/>
                <path class="circle"
                      stroke-dasharray="0, 100"
                      d="M18 2.0845
                         a 15.9155 15.9155 0 0 1 0 31.831
                         a 15.9155 15.9155 0 0 1 0 -31.831"
                      stroke="url(#gradient)"/>
                <text x="18" y="15.35" class="percentage">0%</text>
                <text x="18" y="25.35" class="ai-generated-text">AI Generated</text>
              </svg>
            </div>

            <div class="separator"></div>

            <div class="horizontal-progress">
              <div class="progress-bar">
                <div class="filler empty" style="width: 100%"></div>
              </div>
              <div class="percentage-label">
                <div class="label-group">
                  <span class="dot human-dot"></span><span class="label">0%</span>
                </div>
                <div class="label-group">
                  <span class="dot ai-dot"></span><span class="label">0%</span>
                </div>
              </div>
            </div>

            <div v-if="result === null" class="keywords">
              <span class="keyword-title">关键字：</span>
              <span class="keyword green">100% 人类</span>
              <span class="keyword yellow">50% 人类</span>
              <span class="keyword red">0% 人类</span>
            </div>
            <div v-else class="result-text">
              LLM Text Detector显示文本有 {{ percentage }}% 的概率是虚假的。
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
export default {
  data() {
    return {
      activeTab: 'text',
      file: null,
      textInput: '',
      result: null,
      selectedOption: 'text',
      fileSelected: false
    };
  },
  computed: {
    percentage() {
      return this.result !== null ? (this.result * 100).toFixed(0) : 0;
    },
    humanPercentage() {
      return (100 - this.percentage).toFixed(0);
    }
  },
  methods: {
    handleFileUpload(event) {
      this.file = event.target.files[0];
      this.fileSelected = true;
    },
    async uploadFile(flag = 0) {
    let formData = new FormData();
    formData.append('file', this.file);
    formData.append('flag', flag);  // 将 flag 添加到 formData 中

    try {
      let response = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData
      });
      let result = await response.json();
      this.result = result.result;
      this.downloadProcessedFile(result.processed_file);
      console.log(result);
    } catch (error) {
      console.error('Error:', error);
    }
  },

  async quickUploadFile() {
    this.uploadFile(1);  // 调用 uploadFile，并传递 flag=1
  },
    async downloadProcessedFile(filePath) {
      const filename = filePath.split('/').pop();
      const response = await fetch(`http://127.0.0.1:5000/processed/${filename}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    },
    async checkContent() {
      try {
        let response = await fetch('http://127.0.0.1:6006/check', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: this.textInput })
        });
        let result = await response.json();
        this.result = result.result;
        console.log(result);
      } catch (error) {
        console.error('Error:', error);
      }
    },
    clearContent() {
      this.textInput = '';
    },
    async quickCheckContent() {
    try {
      let response = await fetch('http://127.0.0.1:5000/check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: this.textInput, flag: 1 }) // 传递 flag: 1
      });
      let result = await response.json();
      this.result = result.result;
      console.log(result);
    } catch (error) {
      console.error('Error:', error);
    }
  },
    selectOption(option) {
      this.selectedOption = option;
      this.result = null;
      this.textInput = '';
      this.file = null;
      this.fileSelected = false;
    },
    triggerFileInput() {
      this.$refs.fileInput.click();
    },
    removeFile() {
      this.file = null;
      this.fileSelected = false;
    },
    navigateToHomePage() {
      window.location.href = '/home';
    }
  }
};
</script>

<style>
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
}

.app-container {
  background: #f8f9fa; /* 接近图片中的浅灰色背景 */
  min-height: 100vh;
  min-width: 100vw;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  box-sizing: border-box;
}

.header {
  display: flex;
  align-items: center;
  position: fixed;
  top: 15px;
  left: 35px;
  cursor: pointer;
}

.logo {
  width: 40px;
  height: 40px;
  margin-right: 10px;
}

.gradient-text {
  background: linear-gradient(to right, #8a2be2, #4b0082);
  -webkit-background-clip: text;
  color: transparent;
}

main {
  display: flex;
  width: 100%;
  max-width: 1200px;
  margin-top: 100px;
}

.content-wrapper {
  display: flex;
  width: 100%;
}

.left-container {
  flex: 2;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.right-container {
  flex: 1.5;
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  margin-left: 10px;
  text-align: center;
}

.selection {
  display: flex;
  justify-content: flex-start;
  margin: 20px 0;
}

.selection button {
  margin: 0 10px;
  padding: 10px 20px;
  border: none;
  cursor: pointer;
  background-color: #f0f0f0;
  transition: background-color 0.3s ease, border-radius 0.3s ease;
  border-radius: 8px;
  display: flex;
  align-items: center;
}

.selection button .icon {
  width: 20px;
  height: 20px;
  margin-right: 8px;
}

.selection button.active {
  background-color: #8a2be2;
  color: white;
}

.selection button:hover {
  background-color: #4b0082;
  color: white;
}

.content-container {
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 600px;
  margin-top: 20px;
  height: 400px;
}

.upload-container {
  position: relative;
}

.file-input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.upload-text {
  text-align: center;
  margin-top: 150px;
  font-size: 16px;
  color: #666;
}

.icon-container {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}

.word-icon, .pdf-icon {
  width: 40px;
  height: 40px;
  margin: 0 5px;
}

.file-display-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

.file-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  background: #f0f0f0;
  padding: 10px;
  border-radius: 10px;
  margin-bottom: 20px;
}

.remove-button {
  background: none;
  border: none;
  cursor: pointer;
  color: #333;
  display: flex;
  align-items: center;
  justify-content: center;
}

.remove-icon {
  fill: #333;
  width: 20px;
  height: 20px;
}

.textarea-container {
  position: relative;
  width: 100%;
}

textarea {
  width: calc(100% - 20px);
  padding: 10px;
  margin: 10px 10px 10px 0;
  border: 1px solid #ccc;
  border-radius: 4px;
  resize: vertical;
  height: 300px;
}

.word-count {
  position: absolute;
  bottom: 30px;
  right: 10px;
  color: #666;
  font-size: 12px;
}

.button-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 10px;
}

.button-container button {
  margin-left: 10px;
}

button, .upload-button {
  padding: 10px 20px;
  border: none;
  background-color: #8a2be2;
  color: white;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.3s ease;
}

button:hover, .upload-button:hover {
  background-color: #4b0082;
}

.result-container {
  margin-top: 20px;
}

.circle-progress {
  display: inline-block;
  position: relative;
}

.circular-chart {
  display: block;
  margin: 10px auto;
  width: 200px;
  height: 200px;
}

.circle-bg {
  fill: none;
  stroke: #eee;
  stroke-width: 2.2;
}

.circle {
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  transition: stroke-dasharray 0.3s ease;
}

.percentage {
  fill: #010203;
  font-size: 0.25em;
  text-anchor: middle;
}

.ai-generated-text {
  fill: #565656;
  font-size: 0.18em;
  text-anchor: middle;
}

.separator {
  width: 80%;
  height: 1px;
  background-color: #ccc;
  margin: 20px auto;
}

.horizontal-progress {
  margin-top: 20px;
  position: relative;
}

.progress-bar {
  width: 80%;
  height: 13px;
  background-color: #eee;
  border-radius: 5px;
  margin: 0 auto;
  display: flex;
}

.filler {
  height: 100%;
}

.filler.left {
  background-color: #b5e7a0;
  border-top-left-radius: inherit;
  border-bottom-left-radius: inherit;
}

.filler.right {
  background-color: #8bc4ff;
  border-top-right-radius: inherit;
  border-bottom-right-radius: inherit;
}

.filler.empty {
  background-color: #eee;
  border-radius: inherit;
}

.percentage-label {
  display: flex;
  justify-content: space-between;
  width: 80%;
  margin: 0 auto;
  font-size: 14px;
  color: #010203;
}

.label-group {
  display: flex;
  align-items: center;
}

.dot {
  height: 10px;
  width: 10px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 5px;
}

.human-dot {
  background-color: #b5e7a0;
}

.ai-dot {
  background-color: #8bc4ff;
}

.label {
  margin-left: 5px;
}

.keywords {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  margin-top: 10px;
  width: 80%;
  margin-left: auto;
  margin-right: auto;
}

.keyword-title {
  margin-right: 10px;
  font-size: 16px;
}

.keyword {
  margin-right: 10px;
  font-size: 16px;
}

.keyword.green {
  color: #00FF00;
}

.keyword.yellow {
  color: #FFD700;
}

.keyword.red {
  color: #FF0000;
}

.result-text {
  margin-top: 20px;
  font-size: 16px;
  color: #565656;
  text-align: center;
}
</style>
