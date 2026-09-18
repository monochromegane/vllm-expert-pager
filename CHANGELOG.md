# Changelog

## [0.1.1](https://github.com/monochromegane/vllm-expert-pager/compare/0.1.0...0.1.1) - 2026-09-18

- Perf/ssd tier by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/2
- Keep the RAM and SSD tiers compressed and decode the rows on the GPU. by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/4
- Recover when a fetch request or its done never reaches the other side. by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/5
- Seed the RAM tier with the prefill's most-used experts. by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/6
- Serve the SSD requests from a C thread through a request ring. by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/7
- Read RAM rows through four looping programs; the SM gather doubles to… by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/8
- Decode compressed rows in a separate launch; the fused copy read the … by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/9
- Run the prefill in 64-row chunks of the working slab; the freed VRAM … by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/10
- Re-measure the results on the current code; the default pitch goes to… by @monochromegane in https://github.com/monochromegane/vllm-expert-pager/pull/11

## [v0.1.0](https://github.com/monochromegane/vllm-expert-pager/commits/v0.1.0) - 2026-09-11
