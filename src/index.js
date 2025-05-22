const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

// Serve static files if needed
app.use(express.static(path.join(__dirname, 'public')));

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

app.get('/deepseek', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'deepseek.html'));
});

app.get('/chat', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'chatgpt.html'));
});

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
