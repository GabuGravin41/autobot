FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Set environment
ENV HOST=0.0.0.0
ENV PORT=3000

# Build for production could go here, but for development we just start Vite
EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]
