# GitHub Pages 部署指南

## 🚀 自动部署已配置完成

你的 Youtu-Embedding 文档网站已经配置好了 GitHub Pages 自动部署！

### 📁 已创建的文件

1. **`.github/workflows/deploy-docs.yml`** - GitHub Actions 工作流
2. **`docs/next.config.mjs`** - Next.js 静态导出配置
3. **`docs/public/.nojekyll`** - GitHub Pages 配置文件

### 🔧 配置详情

- **构建目录**: `docs/`
- **输出目录**: `docs/out/`
- **基础路径**: `/Youtu-Embedding` (生产环境)
- **触发条件**: 推送到 `main` 分支时自动部署

### 📋 启用步骤

1. **推送代码到 GitHub**:
   ```bash
   git add .
   git commit -m "Add GitHub Pages deployment"
   git push origin main
   ```

2. **在 GitHub 仓库中启用 Pages**:
   - 进入仓库设置 → Pages
   - Source 选择 "GitHub Actions"
   - 等待第一次部署完成

3. **访问你的文档网站**:
   - URL: `https://你的用户名.github.io/Youtu-Embedding`

### ✅ 功能特性

- ✅ **自动构建**: 每次推送代码自动触发构建
- ✅ **静态导出**: 完全静态的 HTML 文件
- ✅ **GA4 追踪**: 已集成 Google Analytics
- ✅ **自定义 Logo**: 支持 SVG logo
- ✅ **响应式设计**: 移动端友好
- ✅ **MDX 支持**: 支持 Markdown 和 React 组件

### 🛠️ 本地测试

```bash
cd docs
npm run build    # 构建静态文件
npm run start    # 预览构建结果
```

### 📝 注意事项

- 首次部署可能需要几分钟时间
- 确保仓库是公开的，或者有 GitHub Pro 账户
- 如果修改了 basePath，需要同步更新 GitHub Pages 设置

现在你可以推送代码，GitHub Pages 就会自动部署你的文档网站了！🎊