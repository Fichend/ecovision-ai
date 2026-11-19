import type { Metadata } from "next";
import "./globals.css";
import { DeploymentNotice } from "@/components/DeploymentNotice";
import { Footer } from "@/components/Footer";
import { Navigation } from "@/components/Navigation";

export const metadata: Metadata = {
  title: "EcoVision AI",
  description: "Explainable waste recognition and source-grounded recycling guidance."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-scroll-behavior="smooth" suppressHydrationWarning>
      <body>
        <Navigation />
        <DeploymentNotice />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
