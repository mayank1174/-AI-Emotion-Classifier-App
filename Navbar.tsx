import { Moon, Sun, BrainCircuit } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "./button";

export function Navbar() {
  const { theme, setTheme } = useTheme();

  return (
    <nav className="w-full absolute top-0 z-50 bg-transparent">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Logo removed as requested */}
        </div>
        
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-full w-9 h-9 bg-purple-50 hover:bg-purple-100 dark:bg-white/5 dark:hover:bg-white/10"
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0 text-amber-500" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100 text-purple-400" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </div>
    </nav>
  );
}
