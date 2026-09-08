export interface Testimonial {
  id: string;
  name: string;
  role: string;
  company?: string;
  avatar: string;
  companyLogo?: string;
  quote: string;
}

export const testimonials: Testimonial[] = [
  {
    id: "sarah-j",
    name: "Sarah J.",
    role: "Marketing Director",
    company: "LeadScale",
    avatar: "/images/testimonials/customer-sarah.webp",
    quote:
      "The best investment we've made for our marketing team. We went from paying Mailchimp $800/mo to with free core + £50 one-time add-on bundle via early access. The self-hosting was easy and the leads are high quality.",
  },
  {
    id: "marcus-k",
    name: "Marcus K.",
    role: "SaaS Founder",
    company: "CloudPulse",
    avatar: "/images/testimonials/customer-marcus.webp",
    quote:
      "Finally, an email platform that doesn't punish success. Our list grew to 50k contacts and our costs stayed exactly the same. The AI agents saved us hours of manual work every week.",
  },
  {
    id: "chen-l",
    name: "Chen L.",
    role: "CTO",
    company: "DevEngine",
    avatar: "/images/testimonials/customer-chen.webp",
    quote:
      "The AI agents alone are worth 10x the one-time add-on price. They handle our lead warmups and basic inquiries perfectly. Having full source code access is a developer's dream.",
  },
  {
    id: "david",
    name: "David",
    role: "Product Manager",
    company: "SalesBridge",
    avatar: "/images/testimonials/company-salesbridge.webp",
    companyLogo: "/images/testimonials/company-salesbridge.webp",
    quote:
      "The Laravel Mail Platform crew walked us through setup step by step. Their playbook on nurturing and converting leads has been a total game-changer.",
  },
  {
    id: "maria",
    name: "Maria",
    role: "CEO",
    company: "Moldova Digital",
    avatar: "/images/testimonials/company-moldova.webp",
    companyLogo: "/images/testimonials/company-moldova.webp",
    quote:
      "We used to bleed leads in hand-offs. Now with Laravel Mail, we retain, track, and convert like pros. Feels like we built a sales engine in-house.",
  },
  {
    id: "juan-p",
    name: "Juan P.",
    role: "Lead Architect",
    company: "StackFlow",
    avatar: "/images/testimonials/customer-juan.webp",
    quote:
      "Self-hosting Laravel Mail gave us 100% control over our customer email data and IP reputation. The campaign automation workflows integrated directly into our Laravel backend without any fuss.",
  },
];
