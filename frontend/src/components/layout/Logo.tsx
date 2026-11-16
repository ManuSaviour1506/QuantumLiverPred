export function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id="lti-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6366f1" /><stop offset="1" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="#0e1424" stroke="url(#lti-g)" strokeWidth="1.2" />
      <path d="M8.5 19.5c0-6.2 5.2-10.5 11.3-10.5 2 0 3.7 1.2 3.7 3.3 0 7.1-6.3 11.2-11.3 11.2-2.2 0-3.7-1.6-3.7-4z" fill="url(#lti-g)" />
      <circle cx="21" cy="12.5" r="1.6" fill="#e7ecf6" />
    </svg>
  );
}
