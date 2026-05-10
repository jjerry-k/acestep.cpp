<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		open = $bindable(false),
		title,
		body,
		onConfirm
	}: {
		open: boolean;
		title: string;
		body?: Snippet;
		onConfirm: () => void;
	} = $props();

	let root = $state<HTMLDivElement>();

	function cancel() {
		open = false;
	}

	function confirm() {
		open = false;
		onConfirm();
	}

	// Same outside-click pattern as Menu.svelte. Listeners attach only while
	// open so closed dialogs cost nothing globally. Enter confirms, Escape
	// cancels: both work whether focus sits on the buttons or inside an input.
	$effect(() => {
		if (!open) return;
		const onMouseDown = (e: MouseEvent) => {
			if (!root?.contains(e.target as Node)) cancel();
		};
		const onKey = (e: KeyboardEvent) => {
			if (e.key === 'Escape') cancel();
			else if (e.key === 'Enter') confirm();
		};
		document.addEventListener('mousedown', onMouseDown);
		document.addEventListener('keydown', onKey);
		return () => {
			document.removeEventListener('mousedown', onMouseDown);
			document.removeEventListener('keydown', onKey);
		};
	});
</script>

{#if open}
	<div class="dialog-overlay">
		<div class="dialog" bind:this={root}>
			<div class="dialog-title">{title}</div>
			{#if body}
				<div class="dialog-body">{@render body()}</div>
			{/if}
			<div class="dialog-footer">
				<button type="button" class="dialog-btn" onclick={cancel}>Cancel</button>
				<button type="button" class="dialog-btn" onclick={confirm}>OK</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.dialog-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 100;
	}
	.dialog {
		background: var(--bg-card);
		border-radius: 4px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
		padding: 0.75rem;
		min-width: 16rem;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.dialog-title {
		font-size: 0.85rem;
		color: var(--fg);
	}
	.dialog-body {
		font-size: 0.8rem;
		color: var(--fg);
	}
	.dialog-footer {
		display: flex;
		justify-content: flex-end;
		gap: 0.4rem;
	}
	.dialog-btn {
		background: var(--bg-btn);
		border: none;
		cursor: pointer;
		padding: 0.2rem 0.6rem;
		color: var(--fg);
		font-size: 0.8rem;
		border-radius: 3px;
	}
	.dialog-btn:hover {
		background: var(--bg-btn-hover);
	}
</style>
