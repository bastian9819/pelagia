import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettier from 'eslint-config-prettier';

export default tseslint.config(
  {
    ignores: ['dist', 'node_modules', 'coverage'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Hot simulation loops sometimes need plain numeric indexing and casts;
      // keep the strict defaults but allow pragmatic exceptions where flagged.
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },
  // Plain JS config files (this file, etc.) are not part of the typed program,
  // so lint them without type information to avoid project-service errors.
  {
    files: ['**/*.js'],
    extends: [tseslint.configs.disableTypeChecked],
  },
  // Disable formatting-related rules; Prettier owns formatting.
  prettier,
);
