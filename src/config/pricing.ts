export const pricing = {
  core: {
    price: 0,
    currency: 'GBP',
    label: 'Free',
    title: 'Core',
    description: 'Open-source self-hosted engine. Free forever.',
    badge: 'Alpha',
    note: 'Alpha access – request via email.',
    features: ['Self-hosted engine', 'SMTP, queues & notifications', 'Unlimited contacts & sends', 'Full source code'],
  },
  addons: {
    price: 50,
    currency: 'GBP',
    label: '£50 one-time',
    title: 'Add-on Bundle',
    description: 'Leads Database + Validation API credits. One-time fee, lifetime access.',
    badge: 'Alpha',
    note: 'Alpha access – request via email.',
    features: ['Leads Database (18M+ verified B2B leads)', 'Validation API credits', 'One-time fee – no monthly, no per-credit', 'Includes all future add-on updates'],
  },
  earlyAccessEmail: 'info@laravelmail.com',
  earlyAccessSubject: 'Early Access Request - [Your Name]',
  get mailto() {
    return `mailto:${this.earlyAccessEmail}?subject=${encodeURIComponent(this.earlyAccessSubject)}`;
  },
} as const;
