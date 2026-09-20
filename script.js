(() => {
  const stage = document.getElementById('system-stage');
  const art = document.getElementById('orbit-art');
  const worlds = [...document.querySelectorAll('.world')];
  const pages = [...document.querySelectorAll('[data-page]')];
  const research = document.getElementById('research');
  const current = document.getElementById('current-section');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const paths = [...document.querySelectorAll('.text-nav a')];
  const menuToggle = document.getElementById('menu-toggle');
  const menuClose = document.getElementById('menu-close');
  const drawer = document.getElementById('section-drawer');
  const backdrop = document.getElementById('drawer-backdrop');
  const drawerLinks = [...drawer.querySelectorAll('a')];
  const aboutTrigger = document.getElementById('about-spaceship');
  const aboutDialog = document.getElementById('about-dialog');
  const aboutClose = document.getElementById('about-close');
  let previousFocus;
  function setMenu(open, restoreFocus = true) {
    drawer.classList.toggle('is-open', open);
    drawer.inert = !open;
    drawer.setAttribute('aria-hidden', String(!open));
    menuToggle.setAttribute('aria-expanded', String(open));
    menuToggle.setAttribute('aria-label', open ? 'Close section navigator' : 'Open section navigator');
    backdrop.hidden = !open;
    document.body.style.overflow = open ? 'hidden' : '';
    if (open) { previousFocus = document.activeElement; menuClose.focus(); }
    else if (restoreFocus && previousFocus) previousFocus.focus();
  }
  menuToggle.addEventListener('click', () => setMenu(!drawer.classList.contains('is-open')));
  menuClose.addEventListener('click', () => setMenu(false));
  backdrop.addEventListener('click', () => setMenu(false));
  drawerLinks.forEach(link => link.addEventListener('click', () => setMenu(false, false)));
  aboutTrigger.addEventListener('click', () => aboutDialog.showModal());
  aboutClose.addEventListener('click', () => aboutDialog.close());
  aboutDialog.addEventListener('click', event => {
    const box = aboutDialog.getBoundingClientRect();
    const outside = event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom;
    if (outside) aboutDialog.close();
  });
  aboutDialog.addEventListener('close', () => aboutTrigger.focus());
  document.addEventListener('keydown', e => {
    if (!drawer.classList.contains('is-open')) return;
    if (e.key === 'Escape') { setMenu(false); return; }
    if (e.key === 'Tab') {
      const focusable = [menuClose, ...drawerLinks];
      const index = focusable.indexOf(document.activeElement);
      if (e.shiftKey && index <= 0) { e.preventDefault(); focusable.at(-1).focus(); }
      else if (!e.shiftKey && index === focusable.length - 1) { e.preventDefault(); focusable[0].focus(); }
    }
  });
  const start = performance.now();
  let running = false;
  let rx = 0, ry = 0;

  function resize() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    rx = Math.min(width * (width < 700 ? 0.45 : 0.465), height * (width < 700 ? 0.65 : 1.18));
    ry = Math.min(height * (width < 700 ? 0.29 : 0.345), width * 0.43);
    const ns = 'http://www.w3.org/2000/svg';
    art.setAttribute('viewBox', `0 0 ${width} ${height}`);
    art.replaceChildren();
    for (let i = 1; i <= 9; i++) {
      const ring = document.createElementNS(ns, 'ellipse');
      const f = 0.13 + i * 0.087;
      ring.setAttribute('cx', width / 2);
      ring.setAttribute('cy', height * (width < 700 ? 0.59 : 0.61));
      ring.setAttribute('rx', rx * f);
      ring.setAttribute('ry', ry * f);
      art.append(ring);
    }
  }

  function animate(now) {
    const t = (now - start) / 1000;
    worlds.forEach(world => {
      const ring = Number(world.dataset.orbit);
      const f = 0.13 + ring * 0.087;
      const initial = Number(world.dataset.angle) * Math.PI / 180;
      const period = Number(world.dataset.period);
      const angle = initial + (reduced.matches || !ring ? 0 : t * 2 * Math.PI / period);
      const x = stage.clientWidth / 2 + rx * f * Math.cos(angle);
      const y = stage.clientHeight * (stage.clientWidth < 700 ? 0.59 : 0.61) + ry * f * Math.sin(angle);
      world.style.left = `${x}px`;
      world.style.top = `${y}px`;
    });
    if (!reduced.matches && !document.hidden) requestAnimationFrame(animate);
    else running = false;
  }

  function openPage(scroll = true) {
    const name = decodeURIComponent(location.hash.slice(1));
    const page = pages.find(item => item.id === name);
    research.hidden = !page;
    pages.forEach(item => { item.hidden = item !== page; });
    drawerLinks.forEach(link => {
      if (page && link.hash === location.hash) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    paths.forEach(link => {
      if (page && link.hash === location.hash) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    if (page) current.textContent = page.querySelector('h2').textContent;
    if (scroll) (page ? research : document.getElementById('system')).scrollIntoView({behavior: reduced.matches ? 'instant' : 'smooth', block: 'start'});
  }

  resize();
  running = !reduced.matches;
  animate(start);
  addEventListener('resize', () => { resize(); if (reduced.matches) animate(start); });
  addEventListener('visibilitychange', () => { if (!document.hidden && !reduced.matches && !running) { running = true; requestAnimationFrame(animate); } });
  reduced.addEventListener('change', () => { if (!reduced.matches && !running) { running = true; requestAnimationFrame(animate); } else if (reduced.matches) animate(start); });
  addEventListener('hashchange', () => openPage());
  addEventListener('popstate', () => openPage());
  document.getElementById('home-link').addEventListener('click', () => { if (location.hash === '#system') openPage(); });
  document.getElementById('overview').addEventListener('click', () => { if (location.hash === '#system') openPage(); });
  worlds.forEach(world => world.addEventListener('click', () => { if (location.hash === world.hash) openPage(); }));
  openPage(Boolean(location.hash && location.hash !== '#system'));
})();
