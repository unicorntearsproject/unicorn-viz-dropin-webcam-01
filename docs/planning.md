# webcam-01 Planning

Owner: Webcam Drop-in Maintainers
Status: active
Last updated: 2026-06-05

## Purpose

Track webcam drop-in planning items that depend on multi-head behavior,
hotkey architecture, and upcoming control-room API surfaces.

## Current Priority Items

0. Webcam editor modal (new requirement)
   - Add a dedicated webcam editor modal in the runtime/control-room surface.
   - Modal must support camera-device list + per-device enable/disable.
   - Modal must support manual camera rediscovery/rescan.
   - Modal must expose live image controls for at least brightness and
     contrast.
   - Modal must expose horizontal and vertical flip toggles.
   - Modal actions should be routed through public hotkey/control-room
     registration interfaces (no direct private runtime mutation from callers).
   - Status (2026-06-05): complete for current scope.
     - Added camera list + rediscovery + enable/disable APIs.
     - Added brightness/contrast control APIs.
     - Added horizontal/vertical flip control APIs.
     - Added control-room wiring for camera/device/image controls.
     - Added camera-switch auto-hide workaround (temporary until switch lag is
       improved).
     - Added dedicated webcam editor modal wiring (hotkey + routed modal
       snapshot + runtime controls).
     - Added direct camera select path (not only prev/next).
     - Added explicit switching status indicator for operator surfaces.
     - Added global runtime-state persistence for webcam state, including:
       selected camera, disabled cameras, layout/PiP/treatment, and per-camera
       brightness/contrast/flip settings.
   - Decision (2026-06-05): keep-all-cameras-open hot-swap mode is explicitly
     deferred; current policy remains single active camera open at a time.

1. Monitor config modal integration (CR API dependent)
   - Add a monitor-config modal reachable by both:
     - a dedicated hotkey path, and
     - a control-room button using the registration interfaces.
   - Modal must allow changing monitor-related options, including a quick
     include/exclude toggle for a selected monitor.
   - Add a secondary (non-primary) hotkey linkage for the quick include/exclude
     toggle so operators can flip exclusion state without opening the full modal.
   - Keep implementation aligned with hotkey registration and control-room
     registration interfaces; final wiring depends on the forthcoming CR API.

2. Mirror/span behavior expansion and keymap cleanup
   - Status (2026-06-05): base runtime semantics and hotkeys implemented in
     core multi-head/app/hotkey paths; platform validation and any keymap
     refinements remain open.
   - Add explicit mirror and span options that respect excluded displays by
     default (included-display behavior).
   - Redefine any "all" behavior so "all" means true all displays and does not
     respect the exclusion list.
   - Update hotkeys to match the clarified semantics.
   - Evaluate whether one `X`-family combination should cycle display modes to
     avoid operator key overload; finalize after operator review.

3. Span-mode alignment remediation (all platforms)
   - Resolve outstanding alignment issues in span mode across supported
     platforms/compositors.
   - Validate overlay/HUD/menu/splash placement under mixed-size and mixed-DPI
     monitor layouts.
   - Record platform-specific constraints and fallback behavior where compositor
     limits prevent strict placement control.

  4. Monitor editor v1 (control-room surface)
     - Status (2026-06-05): in progress.
     - Added control-room fullscreen monitor editor entry point.
     - Added SDL-detected display topology map rendering.
     - Added display mode buttons wired to supported span/mirror variants.
     - Added include/exclude preference toggles saved to runtime-state for next
       launch.
     - Next: refine UX details based on operator feedback.

## Execution Notes

- Item 1 is blocked pending control-room API availability.
- Items 2 and 3 can proceed in parallel while CR API work is in flight.
- Known issue note (2026-06-05): camera-device switching is stable on Fedora 44,
  and Windows 11 validation pass has been completed for current workflow.
  Status update: root-cause fix applied in webcam worker (`_stop` name
  collision removed) and retested.
- Outstanding webcam-specific items (2026-06-05):
  - Done: fresh Windows 11 operator validation pass for repeated camera
    switching under load.
  - Done: switch-hide timing tuned for current operator workflow.
  - N/A (deferred): guarded prewarm of neighboring cameras behind config
    flags; keep disabled by default on Windows.
- Use the multi-head docs and audits as implementation references:
  - `drop-ins/multi-head-01/docs/operations.md`
  - `drop-ins/multi-head-01/docs/troubleshooting.md`
  - `audits/multi-head-2026-05-10.md`