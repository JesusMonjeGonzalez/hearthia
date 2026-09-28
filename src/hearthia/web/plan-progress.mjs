/* Plan progress and auto-continue decisions (pure, unit-tested). */

export function planProgress(plan) {
  const steps = Array.isArray(plan?.steps) ? plan.steps : [];
  const raw = Array.isArray(plan?.done) ? plan.done : [];
  const doneSet = new Set(
    raw.filter((n) => Number.isInteger(n) && n >= 1 && n <= steps.length),
  );
  return {
    total: steps.length,
    done: doneSet.size,
    pending: steps.length - doneSet.size,
    complete: steps.length > 0 && doneSet.size === steps.length,
    doneSet,
  };
}

export function shouldAutoContinue({ plan, strike, limit, enabled }) {
  if (!enabled || strike >= limit) return false;
  const progress = planProgress(plan);
  return progress.total > 0 && !progress.complete;
}

export function autoContinuePrompt(plan, lastTurn) {
  const progress = planProgress(plan);
  const step = progress.done + 1;
  const edited = Array.isArray(lastTurn?.edited) ? lastTurn.edited.length : 0;
  if (edited && lastTurn.verified === false) {
    return (
      `Continúa con el plan. El turno anterior editó ${edited} archivo(s) sin verificar: ` +
      `ejecuta la comprobación pendiente antes de seguir con el paso ${step}.`
    );
  }
  return `Continúa con el siguiente paso pendiente del plan (paso ${step}).`;
}
