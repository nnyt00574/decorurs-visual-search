import "./globals.css";

export const metadata = {
  title: "Visual Search — DecorUrs",
  description: "Upload a photo of furniture you like and find similar pieces in the DecorUrs catalog.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
