import { Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="w-full border-t border-purple-500/10 bg-white/30 dark:bg-black/30 backdrop-blur-lg mt-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <p className="text-sm text-muted-foreground flex items-center gap-1.5">
          &copy; 2026. All rights reserved.
        </p>
        <div className="flex gap-4 text-sm text-muted-foreground">
          <a href="#" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">About</a>
          <a href="#" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">Privacy</a>
          <a href="#" className="hover:text-purple-600 dark:hover:text-purple-400 transition-colors">GitHub</a>
        </div>
      </div>
    </footer>
  );
}
