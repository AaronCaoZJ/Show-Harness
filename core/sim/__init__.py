"""Simulator-side task glue and rollout runners.

Everything in this package talks to a SIMULATOR; nothing in it touches real hardware.
``core/`` proper keeps the shared pieces (``config``, ``images``, ``episode_logger``,
``v0_types``, ...) and the real-robot runners (``real_runner``, ``mvtoken_runner``,
``dual_runner``, ``teleop*``, ``franka/``, ``piper/``).

RoboLab is wired here as a two-file pair -- a task module (env construction +
observation/success/TCP/gripper accessors + axis probe) and an MVTOKEN rollout runner:

===========  ==========================  ===============================
Simulator    task module                 MVTOKEN runner
===========  ==========================  ===============================
RoboLab      ``robolab_task``            ``mvtoken_robolab_runner``
===========  ==========================  ===============================

Nothing is re-exported here on purpose: importing a simulator's task module pulls that
simulator's heavyweight dependencies in (RoboLab wants its own Python 3.11 venv).
Import the specific module you need::

    from core.sim.robolab_task import make_robolab_task
"""
