# Deployment Guide

## Vercel Deployment

This application is configured for deployment on Vercel with the following setup:

### Files Required for Deployment:
- `vercel.json` - Vercel configuration
- `requirements.txt` - Python dependencies
- `api/main.py` - Serverless FastAPI application
- `.gitignore` - Git ignore rules

### Key Changes for Vercel:
1. **WebSocket to Polling**: Replaced WebSocket real-time updates with HTTP polling (every 2 seconds)
2. **Serverless Architecture**: Adapted for Vercel's serverless functions
3. **Simplified Dependencies**: Removed heavy dependencies like PyTorch, gRPC, etc.
4. **In-Memory State**: Demo state is maintained in memory (resets on cold starts)

### Features Preserved:
- ✅ System Console with real-time logs
- ✅ Training Progress Charts (Loss/Accuracy over epochs)
- ✅ Interactive Demo Controls
- ✅ Cluster Visualization
- ✅ Performance Metrics
- ✅ Responsive Dashboard Design

### Deployment Steps:

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   vercel
   ```

4. **Production Deployment:**
   ```bash
   vercel --prod
   ```

### Environment Variables:
No additional environment variables required for basic deployment.

### Limitations:
- State resets on serverless cold starts
- Polling instead of WebSocket (slight delay in updates)
- Single instance (no true distributed training)
- No persistent storage

### Demo URL:
After deployment, you'll receive a URL like: `https://your-project-name.vercel.app`

The dashboard will be fully functional with all interactive features working!