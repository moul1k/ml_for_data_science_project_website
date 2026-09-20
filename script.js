(() => {
  const stage = document.getElementById('system-stage');
  const art = document.getElementById('orbit-art');
  const worlds = [...document.querySelectorAll('.world')];
  const pages = [...document.querySelectorAll('[data-page]')];
  const research = document.getElementById('research');
  const current = document.getElementById('current-section');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const paths = [...document.querySelectorAll('.text-nav a')];
  const start = performance.now();
  let rx = 0, ry = 0;

  function resize() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    rx = Math.min(width * 0.435, height * 0.94);
    ry = Math.min(height * 0.405, width * 0.49);
    const ns = 'http://www.w3.org/2000/svg';
    art.setAttribute('viewBox', `0 0 ${width} ${height}`);
    art.replaceChildren();
    for (let i = 1; i <= 9; i++) {
      const ring = document.createElementNS(ns, 'ellipse');
      const f = 0.13 + i * 0.087;
      ring.setAttribute('cx', width / 2);
      ring.setAttribute('cy', height / 2);
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
      const y = stage.clientHeight / 2 + ry * f * Math.sin(angle);
      world.style.left = `${x}px`;
      world.style.top = `${y}px`;
    });
    if (!reduced.matches && !document.hidden) requestAnimationFrame(animate);
  }

  function openPage(scroll = true) {
    const name = decodeURIComponent(location.hash.slice(1));
    const page = pages.find(item => item.id === name);
    research.hidden = !page;
    pages.forEach(item => { item.hidden = item !== page; });
    paths.forEach(link => {
      if (page && link.hash === location.hash) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    if (page) current.textContent = page.querySelector('h2').textContent;
    if (scroll) (page ? research : document.getElementById('system')).scrollIntoView({behavior: reduced.matches ? 'instant' : 'smooth', block: 'start'});
  }

  resize();
  animate(start);
  addEventListener('resize', () => { resize(); if (reduced.matches) animate(start); });
  addEventListener('visibilitychange', () => { if (!document.hidden && !reduced.matches) requestAnimationFrame(animate); });
  reduced.addEventListener('change', () => { if (!reduced.matches) requestAnimationFrame(animate); else animate(start); });
  addEventListener('hashchange', () => openPage());
  addEventListener('popstate', () => openPage());
  document.getElementById('home-link').addEventListener('click', () => { if (location.hash === '#system') openPage(); });
  document.getElementById('overview').addEventListener('click', () => { if (location.hash === '#system') openPage(); });
  worlds.forEach(world => world.addEventListener('click', () => { if (location.hash === world.hash) openPage(); }));
  openPage(Boolean(location.hash && location.hash !== '#system'));
})();
