
// Globally resolve all map GIF files from the assets directory
// This ensures Vite correctly processes and bundles these assets
const mapGlobs = import.meta.glob('../assets/maps/*.gif', { eager: true, as: 'url' });

export function getMapSrc(role: 'director' | 'matcher', mapNum: number): string {
    const suffix = role === 'director' ? 'g' : 'f';
    // Construct the expected relative path key for the glob map
    // Note: glob keys are relative to THIS file
    const filename = `map${mapNum}${suffix}.gif`;

    // We need to match the key format from import.meta.glob
    // The glob is relative to this file? No, relative to pattern. 
    // Wait, if this file is in src/utils, and assets are in src/assets
    // Path: ../assets/maps/*.gif
    const key = `../assets/maps/${filename}`;

    const resolved = mapGlobs[key];

    if (!resolved) {
        console.error(`[MapAssets] Map not found for key: ${key}. Available keys:`, Object.keys(mapGlobs));
        // Return a placeholder or the broken key to make it obvious
        return '';
    }

    return resolved;
}
