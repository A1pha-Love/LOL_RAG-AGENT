/**
 * 前端入口：创建应用实例并挂载到 index.html 的 `#app` 上。
 *
 * 这是整个界面的起点，只做三件事：引入根组件、引入全局主题样式、挂载。
 * 项目里没有引入 vue-router（只有一个页面）也没有引入 Pinia/Vuex
 * （状态由 `composables/useAgent.ts` 的模块级单例承担），所以入口文件很短。
 */

import { createApp } from 'vue'
import App from './App.vue'
// 全局主题（CSS 变量、面板通用样式、滚动条）。组件内的 <style scoped> 依赖这里
// 定义的 var(--surface) 等变量，因此必须在使用任何组件之前引入。
import './styles/main.css'

// 单页应用直接挂载到宿主元素；main.css 里给 #app 设了 height: 100%，
// 三栏布局的高度链（100vh → .shell）依赖它。
createApp(App).mount('#app')
