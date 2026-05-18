# WMP

This context covers the project's locomotion training approaches and the terms used to distinguish its existing control path from experimental alternatives.

## Language

**WMP**:
The project's primary locomotion approach, where a world model supplies latent features to a policy trained on real environment rollouts.
_Avoid_: Dreamer policy, imagination control

**Dreamer Branch**:
An experimental training path that evaluates imagination-based control separately from the primary WMP approach.
_Avoid_: WMP v2, replacement WMP

**Imagined Phase**:
The part of Dreamer Branch training that rolls out latent dynamics and updates behavior without direct access to real-environment-only signals.
_Avoid_: Real rollout, simulator phase

**Chunk Step**:
One world-model time step that aggregates multiple environment steps under the current `wm_update_interval` schedule.
_Avoid_: Env step, frame step

**Latent Critic**:
The primary value estimator for the Dreamer Branch, trained on latent features and used inside the Imagined Phase.
_Avoid_: Privileged critic, environment critic

**Mode Switch**:
An explicit configuration or CLI switch that selects between the original WMP path and the Dreamer Branch without code edits.
_Avoid_: Manual code change, implicit branch

## Relationships

- **WMP** is the primary training approach in this project
- A **Dreamer Branch** is evaluated against **WMP**, not defined as a rewrite of it
- The **Imagined Phase** belongs to the **Dreamer Branch** and excludes real-environment-only signals such as current AMP transitions and privileged observations
- A **Chunk Step** spans multiple env steps and is the time unit for the first Dreamer Branch replay design
- The **Latent Critic** is the primary critic in the Dreamer Branch, while any privileged critic is limited to support roles outside the **Imagined Phase**
- A **Mode Switch** must keep the original WMP path intact and make Dreamer Branch changes reversible without editing code

## Example dialogue

> **Dev:** "Are we replacing **WMP** with this new method?"
> **Domain expert:** "No. The **Dreamer Branch** is a separate experiment that we compare against **WMP**."

> **Dev:** "Can **AMP** and the privileged critic stay in the **Dreamer Branch**?"
> **Domain expert:** "Yes, but not inside the **Imagined Phase**. They stay on the real-rollout or bootstrap side."

> **Dev:** "When we say horizon 16, is that 16 simulator steps?"
> **Domain expert:** "No. In the first Dreamer Branch design, that means 16 **Chunk Steps**, not 16 env steps."

> **Dev:** "Which critic is the main one in the Dreamer Branch?"
> **Domain expert:** "The **Latent Critic**. Any privileged critic only supports real-rollout supervision or bootstrap outside the **Imagined Phase**."

> **Dev:** "Can we just mix both methods in one implementation and manually comment branches?"
> **Domain expert:** "No. Use a **Mode Switch**. Prefer separate runners and modules; if code must be shared in one file, guard it with an explicit switch."

## Flagged ambiguities

- "DreamerV3 version" was used ambiguously to mean both a replacement for **WMP** and a separate **Dreamer Branch** — resolved: it means a separate **Dreamer Branch**
- "Keep AMP / privileged critic" was used ambiguously to mean both "keep them somewhere in the branch" and "use them inside the **Imagined Phase**" — resolved: keep them in the branch, but not inside the **Imagined Phase**
- "step" was used ambiguously to mean both env-step and **Chunk Step** — resolved: first Dreamer Branch replay uses **Chunk Step** as its time unit
- "critic" was used ambiguously to mean both the main imagined-phase critic and a real-rollout support critic — resolved: the main critic is the **Latent Critic**
- "easy to switch back" was underspecified — resolved: switching must happen through an explicit **Mode Switch**, not by editing code paths manually
