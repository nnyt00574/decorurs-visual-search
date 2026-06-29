"use client";

import { useState, useRef } from "react";
import styles from "./page.module.css";

const API_URL = `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/search`;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

interface Product {
  product_id: number;
  name: string;
  image_url: string;
  product_url: string;
  price: string;
  material: string;
  score: number;
}

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<Product[] | null>(null);
  const [queryMaterial, setQueryMaterial] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);
    setResults(null);
    setQueryMaterial(null);

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("That file didn't make it through. Use a JPG or PNG image.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("That file is too large. Keep it under 10MB.");
      return;
    }

    setPreview(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || "Search failed. Try again.");
      }
      const data = await res.json();
      setResults(data.results);
      setQueryMaterial(data.query_material);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <main className={styles.page}>
      <p className={styles.eyebrow}>DecorUrs · Visual Search</p>
      <h1 className={styles.headline}>Find it by sight, not by name</h1>
      <p className={styles.subhead}>
        Upload a photo of a table, top, or base you love. We&apos;ll match it against the catalog by
        how it looks — shape, material, finish — not by what it&apos;s called.
      </p>

      <div
        className={`${styles.dropzone} ${styles.dropzoneTickRight} ${dragActive ? styles.dropzoneActive : ""}`}
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
      >
        {preview ? (
          <img src={preview} alt="Your uploaded photo" className={styles.previewImg} />
        ) : (
          <p className={styles.dropzoneLabel}>
            <strong>Drop a photo here</strong>, or click to choose one
            <br />
            JPG or PNG, up to 10MB
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          className={styles.hiddenInput}
          onChange={onInputChange}
        />
      </div>

      <div className={styles.statusRow}>
        {loading && <p className={styles.statusLoading}>Matching against the catalog…</p>}
        {error && <p className={styles.statusError}>{error}</p>}
      </div>

      {results && results.length === 0 && (
        <p className={styles.emptyState}>
          No close matches in the catalog yet. Try a clearer photo, or one taken from a different angle.
        </p>
      )}

      {results && results.length > 0 && (
        <>
          <h2 className={styles.resultsHeading}>
            Closest matches
            {queryMaterial && <span className={styles.detectedMaterial}> — detected as {queryMaterial}</span>}
          </h2>
          <div className={styles.grid}>
            {results.map((p) => (
              <a
                key={p.product_id}
                href={p.product_url}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.card}
              >
                <div className={styles.cardImageWrap}>
                  <img src={p.image_url} alt={p.name} className={styles.cardImage} />
                </div>
                <div className={styles.cardBody}>
                  <p className={styles.cardName}>{p.name}</p>
                  <p className={styles.cardMaterial}>{p.material}</p>
                  <div className={styles.cardMeta}>
                    <p className={styles.cardPrice}>${p.price} CAD</p>
                    <span className={styles.matchTag}>{(p.score * 100).toFixed(0)}% match</span>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
