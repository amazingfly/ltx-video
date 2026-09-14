# Repository boundaries

- **ltx-video**: still images + an audio track → manifest, generated clips, assembled
  video, and selected shorts. The original `ltx_music_video` and
  `antigravityPicker` import/command names are retained for compatibility.
- **sa3**: audio generation, queues, audio library and review utilities.
- **images**: image generation/training, storybook rendering and visual validation.
- **media-pipeline**: orchestration, dataset/track selection, prompt provenance,
  publishing, and configuration tying the three components together.

Each repository owns its dependencies and local runtime environment. Large media,
model files, credentials, and existing outputs stay on disk and out of Git.
Exchange explicit filesystem paths and JSON manifests, not imports from sibling
checkouts. Component commands must run using their own Python environment.

During migration, ignored image and orchestration paths in the original ltxVideo
checkout remain available. They are transferred when their destination is handled;
compatibility links preserve old launch paths. The old copied SA3 Colab example is
preserved locally under `.local/reorganization/sa3ColabExample` for comparison at
the SA3 step. Original source and Git-history backups are in that same directory.
