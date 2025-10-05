import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // 👇 এই 'define' ব্লকটি Node.js-এর 'process' সমস্যা সমাধান করবে
  define: {
    'process.env.NODE_ENV': JSON.stringify('development'),
    // কখনও কখনও পুরো process অবজেক্টটি shim করার প্রয়োজন হতে পারে
    'process': {} 
  }
});