# DeepFlow

DeepFlow is a node-based neural network builder designed for intuitive deep learning model construction. It features a clean, Neubrutalist UI inspired by Heuristica, allowing users to visually assemble and configure neural network architectures.

## Features

- **Visual Node Editor**: Drag-and-drop interface (now click-to-add) for creating neural network layers.
- **Neubrutalist Design**: High-contrast, bold aesthetics with hard shadows and clean typography.
- **Interactive Sidebar**: 
  - Searchable node palette.
  - Categorized blocks (Input, Dense, Conv2D, etc.).
  - Custom node support.
- **React Flow Integration**: Built on top of React Flow for robust node and edge management.
- **Tailwind CSS Styling**: Fully styled using Tailwind CSS for consistent and maintainable design.

## Tech Stack

- **Framework**: React + Vite
- **Styling**: Tailwind CSS
- **Flow Library**: React Flow (@xyflow/react)
- **Icons**: Lucide React

## Getting Started

1.  **Install dependencies**:
    ```bash
    npm install
    ```

2.  **Run the development server**:
    ```bash
    npm run dev
    ```

3.  **Build for production**:
    ```bash
    npm run build
    ```

## Project Structure

- `src/components/`: Core React components (Canvas, Sidebar, RightSidebar, Block, etc.).
- `src/index.css`: Global styles and Tailwind configuration.
- `src/App.jsx`: Main application entry point.
