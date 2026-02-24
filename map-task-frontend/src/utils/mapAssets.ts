
// Globally resolve all map GIF files from the assets directory
// This ensures Vite correctly processes and bundles these assets
const mapGlobs: Record<string, string> = import.meta.glob('../assets/maps/*.gif', { eager: true, query: '?url', import: 'default' }) as any;

export function getMapSrc(role: 'director' | 'matcher', mapNum: number): string {
    const suffix = role === 'director' ? 'g' : 'f';
    const filename = `map${mapNum}${suffix}.gif`;
    const key = `../assets/maps/${filename}`;

    const resolved = mapGlobs[key];

    if (!resolved) {
        console.error(`[MapAssets] Map not found for key: ${key}. Available keys:`, Object.keys(mapGlobs));
        return '';
    }

    return resolved;
}
