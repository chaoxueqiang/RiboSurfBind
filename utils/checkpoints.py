"""Validate evaluation provenance before any benchmark data are loaded."""


def evaluation_set(config, protocol, requested_set=None, checkpoint_config=None):
    if config.get('protocol') != protocol:
        raise ValueError('Checkpoint protocol is %r, not %r; use the matching benchmark protocol'
                         % (config.get('protocol'), protocol))
    stored_set = int(config.get('set_id', 1))
    selected = stored_set if requested_set is None else requested_set
    if protocol == 'hariboss':
        if stored_set not in (1, 2, 3, 4) or selected != stored_set:
            raise ValueError('HARIBOSS checkpoint belongs to Set%d, not Set%d' % (stored_set, selected))
    elif selected != 1 or stored_set != 1:
        raise ValueError('--set applies only to HARIBOSS; other protocols use set_id=1')
    if checkpoint_config is not None:
        for field in ('protocol', 'set_id', 'seed'):
            if config.get(field) != checkpoint_config.get(field):
                raise ValueError('Explicit configuration disagrees with checkpoint metadata: ' + field)
    return selected
