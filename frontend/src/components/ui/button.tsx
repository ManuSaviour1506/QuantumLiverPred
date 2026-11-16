import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef, type ButtonHTMLAttributes } from "react";
import { LuLoaderCircle } from "react-icons/lu";
import { cn } from "../../lib/utils";

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors " +
  "disabled:pointer-events-none disabled:opacity-50 cursor-pointer select-none",
  {
    variants: {
      variant: {
        primary: "bg-brand-600 text-white shadow-sm shadow-brand-900/30 hover:bg-brand-500 dark:bg-brand-500 dark:hover:bg-brand-400",
        navy: "bg-navy-900 text-white hover:bg-navy-800 dark:bg-navy-200 dark:text-navy-950 dark:hover:bg-navy-100",
        outline: "border border-line bg-panel text-ink hover:bg-surface",
        ghost: "text-ink hover:bg-black/5 dark:hover:bg-white/5",
        danger: "bg-red-600 text-white hover:bg-red-700",
        link: "text-brand-700 underline-offset-4 hover:underline dark:text-brand-300 px-0",
      },
      size: { sm: "h-8 px-3", md: "h-10 px-4", lg: "h-11 px-6 text-base", icon: "h-9 w-9" },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, children, disabled, ...props }, ref) => (
    <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} disabled={disabled || loading}
      aria-busy={loading || undefined} {...props}>
      {loading && <LuLoaderCircle className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  ),
);
Button.displayName = "Button";
